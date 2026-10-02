package com.thesis.t2520738

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.Bundle
import android.view.Surface
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var detector: YoloDetector
    private lateinit var overlay: OverlayView
    private lateinit var monitor: TextView
    private lateinit var preview: PreviewView
    private lateinit var videoImage: ImageView

    private val executor =
        Executors.newSingleThreadExecutor()

    private var acceleratorStatus = "OFF"

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        detector = YoloDetector(this)

        overlay =
            findViewById(R.id.overlay)

        monitor =
            findViewById(R.id.monitor)

        preview =
            findViewById(R.id.preview)

        videoImage =
            findViewById(R.id.videoImage)

        preview.scaleType =
            PreviewView.ScaleType.FIT_CENTER

        videoImage.scaleType =
            ImageView.ScaleType.FIT_CENTER

        updateStatus()

        /*
         * ==========================================================
         * CPU BUTTON
         * ==========================================================
         */

        findViewById<Button>(
            R.id.cpuButton
        ).setOnClickListener {

            executor.execute {

                try {

                    detector.setMode(false)

                    acceleratorStatus = "OFF"

                } catch (_: Exception) {

                    acceleratorStatus = "OFF"
                }

                runOnUiThread {
                    updateStatus()
                }
            }
        }

        /*
         * ==========================================================
         * NPU BUTTON
         * ==========================================================
         */

        findViewById<Button>(
            R.id.npuButton
        ).setOnClickListener {

            /*
             * For this thesis build we only identify the
             * validated experimental Samsung device families.
             *
             * Galaxy A35:
             * SM-A356...
             *
             * Galaxy S25 FE:
             * SM-S731...
             */

            if (!isSupportedNpuDevice()) {

                acceleratorStatus = "NOT AVAILABLE"

                executor.execute {

                    try {
                        detector.setMode(false)
                    } catch (_: Exception) {
                    }

                    runOnUiThread {

                        updateStatus()

                        Toast.makeText(
                            this,
                            "Your phone doesn't have a supported NPU.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

                return@setOnClickListener
            }

            /*
             * Device is one of our supported NPU targets.
             *
             * Now initialize the ONNX Runtime NNAPI
             * execution provider.
             */

            executor.execute {

                try {

                    acceleratorStatus =
                        detector.setMode(true)

                } catch (_: Exception) {

                    acceleratorStatus = "FAILED"

                    try {
                        detector.setMode(false)
                    } catch (_: Exception) {
                    }
                }

                runOnUiThread {

                    updateStatus()

                    if (
                        acceleratorStatus ==
                        "FAILED"
                    ) {

                        Toast.makeText(
                            this,
                            "NPU is supported, but NNAPI initialization failed.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }
        }

        /*
         * ==========================================================
         * CAMERA BUTTON
         * ==========================================================
         */

        findViewById<Button>(
            R.id.cameraButton
        ).setOnClickListener {

            startCamera()
        }

        /*
         * ==========================================================
         * VIDEO BUTTON
         * ==========================================================
         */

        findViewById<Button>(
            R.id.videoButton
        ).setOnClickListener {

            pickVideo()
        }

        /*
         * ==========================================================
         * CAMERA PERMISSION
         * ==========================================================
         */

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) ==
            PackageManager.PERMISSION_GRANTED
        ) {

            /*
             * Waiting until PreviewView has completed layout
             * helps prevent the startup issue where camera
             * analysis only started correctly after rotation.
             */

            preview.post {

                if (!isFinishing) {
                    startCamera()
                }
            }

        } else {

            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.CAMERA
                ),
                CAMERA_PERMISSION_CODE
            )
        }
    }

    /*
     * ==============================================================
     * SUPPORTED NPU DEVICE CHECK
     * ==============================================================
     */

    private fun isSupportedNpuDevice(): Boolean {

        val manufacturer =
            Build.MANUFACTURER
                .trim()
                .lowercase()

        val model =
            Build.MODEL
                .trim()
                .uppercase()

        /*
         * We are intentionally strict here.
         *
         * This prevents ordinary phones where NNAPI itself
         * initializes from being incorrectly displayed as
         * having our supported NPU.
         */

        if (
            manufacturer != "samsung"
        ) {
            return false
        }

        /*
         * Samsung Galaxy A35 5G
         *
         * Model family:
         * SM-A356...
         */

        val galaxyA35 =
            model.startsWith(
                "SM-A356"
            )

        /*
         * Samsung Galaxy S25 FE
         *
         * Model family:
         * SM-S731...
         */

        val galaxyS25FE =
            model.startsWith(
                "SM-S731"
            )

        return galaxyA35 ||
            galaxyS25FE
    }

    /*
     * ==============================================================
     * STATUS DISPLAY
     * ==============================================================
     *
     * No:
     *
     * FPS
     * inference time
     * CPU utilization
     * object count
     */

    private fun updateStatus() {

        monitor.text =
            when {

                detector.mode == "NNAPI" &&
                    acceleratorStatus == "ACTIVE" -> {

                    """
                    MODE: NPU
                    NPU: ACTIVE
                    """.trimIndent()
                }

                !isSupportedNpuDevice() -> {

                    """
                    MODE: CPU
                    NPU: NOT AVAILABLE
                    """.trimIndent()
                }

                acceleratorStatus ==
                    "FAILED" -> {

                    """
                    MODE: CPU
                    NPU: FAILED
                    """.trimIndent()
                }

                else -> {

                    """
                    MODE: CPU
                    NPU: AVAILABLE
                    """.trimIndent()
                }
            }
    }

    /*
     * ==============================================================
     * CAMERA
     * ==============================================================
     */

    private fun startCamera() {

        overlay.clear()

        videoImage.visibility =
            View.GONE

        preview.visibility =
            View.VISIBLE

        /*
         * PreviewView must already have dimensions.
         */

        if (
            preview.width <= 0 ||
            preview.height <= 0
        ) {

            preview.post {

                if (!isFinishing) {
                    startCamera()
                }
            }

            return
        }

        val providerFuture =
            ProcessCameraProvider
                .getInstance(this)

        providerFuture.addListener({

            try {

                val cameraProvider =
                    providerFuture.get()

                val rotation =
                    preview.display?.rotation
                        ?: Surface.ROTATION_0

                /*
                 * CAMERA PREVIEW
                 */

                val cameraPreview =
                    Preview.Builder()
                        .setTargetRotation(
                            rotation
                        )
                        .build()

                cameraPreview
                    .setSurfaceProvider(
                        preview.surfaceProvider
                    )

                /*
                 * CAMERA ANALYSIS
                 */

                val analysis =
                    ImageAnalysis.Builder()
                        .setTargetRotation(
                            rotation
                        )
                        .setBackpressureStrategy(
                            ImageAnalysis
                                .STRATEGY_KEEP_ONLY_LATEST
                        )
                        .build()

                analysis.setAnalyzer(
                    executor
                ) { image ->

                    try {

                        /*
                         * CameraX tells us how much rotation
                         * is required for this frame.
                         */

                        val rotationDegrees =
                            image.imageInfo
                                .rotationDegrees

                        val rawBitmap =
                            image.toBitmap()

                        val bitmap =
                            rotateBitmap(
                                rawBitmap,
                                rotationDegrees
                            )

                        runDetection(
                            bitmap
                        )

                        if (
                            bitmap !== rawBitmap
                        ) {

                            rawBitmap.recycle()
                        }

                    } catch (_: Exception) {

                        /*
                         * Never allow one camera frame error
                         * to stop the analyzer.
                         */

                    } finally {

                        image.close()
                    }
                }

                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    cameraPreview,
                    analysis
                )

            } catch (_: Exception) {

                runOnUiThread {

                    monitor.text =
                        """
                        MODE: ${detector.mode}
                        CAMERA: FAILED
                        """.trimIndent()
                }
            }

        }, ContextCompat.getMainExecutor(this))
    }

    /*
     * ==============================================================
     * BITMAP ROTATION
     * ==============================================================
     */

    private fun rotateBitmap(
        source: Bitmap,
        degrees: Int
    ): Bitmap {

        if (
            degrees == 0
        ) {
            return source
        }

        val matrix =
            Matrix().apply {

                postRotate(
                    degrees.toFloat()
                )
            }

        return Bitmap.createBitmap(
            source,
            0,
            0,
            source.width,
            source.height,
            matrix,
            true
        )
    }

    /*
     * ==============================================================
     * DETECTION
     * ==============================================================
     */

    private fun runDetection(
        bitmap: Bitmap
    ) {

        try {

            val detections =
                detector.detect(
                    bitmap
                )

            val sourceWidth =
                bitmap.width

            val sourceHeight =
                bitmap.height

            runOnUiThread {

                overlay.setDetections(
                    detections,
                    sourceWidth,
                    sourceHeight
                )
            }

        } catch (_: Exception) {

            /*
             * One failed frame should not stop
             * camera/video processing.
             */
        }
    }

    /*
     * ==============================================================
     * VIDEO PICKER
     * ==============================================================
     */

    private fun pickVideo() {

        val intent =
            Intent(
                Intent.ACTION_OPEN_DOCUMENT
            ).apply {

                type = "video/*"

                addCategory(
                    Intent.CATEGORY_OPENABLE
                )
            }

        startActivityForResult(
            intent,
            VIDEO_REQUEST_CODE
        )
    }

    /*
     * ==============================================================
     * VIDEO PROCESSING
     * ==============================================================
     */

    @Deprecated(
        "Deprecated in Java"
    )
    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {

        super.onActivityResult(
            requestCode,
            resultCode,
            data
        )

        if (
            requestCode !=
            VIDEO_REQUEST_CODE ||
            resultCode !=
            RESULT_OK
        ) {
            return
        }

        val uri =
            data?.data
                ?: return

        preview.visibility =
            View.GONE

        videoImage.visibility =
            View.VISIBLE

        overlay.clear()

        executor.execute {

            val retriever =
                MediaMetadataRetriever()

            try {

                retriever.setDataSource(
                    this,
                    uri
                )

                val duration =
                    retriever
                        .extractMetadata(
                            MediaMetadataRetriever
                                .METADATA_KEY_DURATION
                        )
                        ?.toLongOrNull()
                        ?: 0L

                val videoRotation =
                    retriever
                        .extractMetadata(
                            MediaMetadataRetriever
                                .METADATA_KEY_VIDEO_ROTATION
                        )
                        ?.toIntOrNull()
                        ?: 0

                var timeMs = 0L

                while (
                    timeMs <
                    duration &&
                    !Thread.currentThread()
                        .isInterrupted
                ) {

                    val rawBitmap =
                        retriever
                            .getFrameAtTime(
                                timeMs *
                                    1000L,
                                MediaMetadataRetriever
                                    .OPTION_CLOSEST
                            )

                    if (
                        rawBitmap == null
                    ) {

                        timeMs +=
                            VIDEO_FRAME_INTERVAL_MS

                        continue
                    }

                    val bitmap =
                        rotateBitmap(
                            rawBitmap,
                            videoRotation
                        )

                    runOnUiThread {

                        videoImage
                            .setImageBitmap(
                                bitmap
                            )
                    }

                    runDetection(
                        bitmap
                    )

                    if (
                        bitmap !== rawBitmap
                    ) {

                        /*
                         * Do not recycle bitmap because the
                         * ImageView currently references it.
                         *
                         * rawBitmap is safe to release when
                         * a separate rotated bitmap exists.
                         */

                        rawBitmap.recycle()
                    }

                    timeMs +=
                        VIDEO_FRAME_INTERVAL_MS
                }

            } catch (_: Exception) {

                runOnUiThread {

                    monitor.text =
                        """
                        MODE: ${detector.mode}
                        VIDEO: FAILED
                        """.trimIndent()
                }

            } finally {

                try {
                    retriever.release()
                } catch (_: Exception) {
                }
            }
        }
    }

    /*
     * ==============================================================
     * CAMERA PERMISSION RESULT
     * ==============================================================
     */

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {

        super.onRequestPermissionsResult(
            requestCode,
            permissions,
            grantResults
        )

        if (
            requestCode ==
            CAMERA_PERMISSION_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] ==
            PackageManager.PERMISSION_GRANTED
        ) {

            preview.post {

                if (!isFinishing) {
                    startCamera()
                }
            }
        }
    }

    /*
     * ==============================================================
     * CLEANUP
     * ==============================================================
     */

    override fun onDestroy() {

        executor.shutdownNow()

        super.onDestroy()
    }

    companion object {

        private const val
            CAMERA_PERMISSION_CODE = 7

        private const val
            VIDEO_REQUEST_CODE = 9

        private const val
            VIDEO_FRAME_INTERVAL_MS = 100L
    }
}
