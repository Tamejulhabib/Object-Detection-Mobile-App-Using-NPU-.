package com.thesis.t2520738

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.view.Surface
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
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

    private lateinit var detector:
        YoloDetector

    private lateinit var overlay:
        OverlayView

    private lateinit var monitor:
        TextView

    private lateinit var preview:
        PreviewView

    private lateinit var videoImage:
        ImageView

    private val executor =
        Executors.newSingleThreadExecutor()

    private var acceleratorStatus =
        "OFF"

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(
            savedInstanceState
        )

        setContentView(
            R.layout.activity_main
        )

        detector =
            YoloDetector(this)

        overlay =
            findViewById(
                R.id.overlay
            )

        monitor =
            findViewById(
                R.id.monitor
            )

        preview =
            findViewById(
                R.id.preview
            )

        videoImage =
            findViewById(
                R.id.videoImage
            )

        /*
         * Important for predictable overlay mapping.
         */
        preview.scaleType =
            PreviewView.ScaleType.FIT_CENTER

        videoImage.scaleType =
            ImageView.ScaleType.FIT_CENTER

        updateStatus()

        findViewById<Button>(
            R.id.cpuButton
        ).setOnClickListener {

            executor.execute {

                try {

                    detector.setMode(false)

                    acceleratorStatus =
                        "OFF"

                } catch (_: Exception) {
                    acceleratorStatus =
                        "OFF"
                }

                runOnUiThread {
                    updateStatus()
                }
            }
        }

        findViewById<Button>(
            R.id.npuButton
        ).setOnClickListener {

            executor.execute {

                try {

                    acceleratorStatus =
                        detector.setMode(true)

                } catch (_: Exception) {

                    acceleratorStatus =
                        "FAILED"
                }

                runOnUiThread {
                    updateStatus()
                }
            }
        }

        findViewById<Button>(
            R.id.cameraButton
        ).setOnClickListener {

            startCamera()
        }

        findViewById<Button>(
            R.id.videoButton
        ).setOnClickListener {

            pickVideo()
        }

        if (
            ContextCompat
                .checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) ==
            PackageManager.PERMISSION_GRANTED
        ) {

            /*
             * Wait until PreviewView has actually
             * completed its first layout.
             *
             * This fixes the behaviour where rotating
             * the phone made the camera suddenly work.
             */
            preview.post {
                startCamera()
            }

        } else {

            ActivityCompat
                .requestPermissions(
                    this,
                    arrayOf(
                        Manifest.permission.CAMERA
                    ),
                    CAMERA_PERMISSION_CODE
                )
        }
    }

    private fun updateStatus() {

        monitor.text =
            if (
                detector.mode ==
                "NNAPI"
            ) {

                """
                MODE: NNAPI
                NPU: $acceleratorStatus
                """.trimIndent()

            } else {

                """
                MODE: CPU
                NPU: OFF
                """.trimIndent()
            }
    }

    private fun startCamera() {

        overlay.clear()

        videoImage.visibility =
            View.GONE

        preview.visibility =
            View.VISIBLE

        /*
         * Do not bind until PreviewView
         * has valid dimensions.
         */
        if (
            preview.width == 0 ||
            preview.height == 0
        ) {

            preview.post {
                startCamera()
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
                         * CameraX provides the amount
                         * the frame must be rotated to
                         * match the display orientation.
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
                            bitmap !==
                            rawBitmap
                        ) {
                            rawBitmap.recycle()
                        }

                    } catch (_: Exception) {

                        /*
                         * Do not allow one bad frame
                         * to kill camera analysis.
                         */

                    } finally {

                        image.close()
                    }
                }

                cameraProvider
                    .unbindAll()

                cameraProvider
                    .bindToLifecycle(
                        this,
                        CameraSelector
                            .DEFAULT_BACK_CAMERA,
                        cameraPreview,
                        analysis
                    )

            } catch (e: Exception) {

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
             * Keep camera/video running
             * if a single inference fails.
             */
        }
    }

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
            data?.data ?: return

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
                    timeMs < duration
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

                    /*
                     * Do not recycle bitmap here because
                     * ImageView is currently displaying it.
                     */

                    timeMs +=
                        VIDEO_FRAME_INTERVAL_MS
                }

            } catch (e: Exception) {

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
                startCamera()
            }
        }
    }

    override fun onDestroy() {

        super.onDestroy()

        executor.shutdown()
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
