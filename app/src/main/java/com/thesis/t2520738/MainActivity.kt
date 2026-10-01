package com.thesis.t2520738

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.os.Process
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

    private lateinit var detector: YoloDetector
    private lateinit var overlay: OverlayView
    private lateinit var monitor: TextView
    private lateinit var preview: PreviewView
    private lateinit var videoImage: ImageView

    private val executor = Executors.newSingleThreadExecutor()

    private var lastFrameTime = System.nanoTime()
    private var previousCpuTime = Process.getElapsedCpuTime()
    private var previousWallTime = System.currentTimeMillis()

    private var acceleratorStatus = "OFF"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        detector = YoloDetector(this)
        detector.setMode(false)

        overlay = findViewById(R.id.overlay)
        monitor = findViewById(R.id.monitor)
        preview = findViewById(R.id.preview)
        videoImage = findViewById(R.id.videoImage)

        findViewById<Button>(R.id.cpuButton).setOnClickListener {
            executor.execute {
                detector.setMode(false)
                acceleratorStatus = "OFF"
            }
        }

        findViewById<Button>(R.id.npuButton).setOnClickListener {
            executor.execute {
                try {
                    acceleratorStatus = detector.setMode(true)
                } catch (e: Exception) {
                    acceleratorStatus = "NNAPI FAILED"
                    detector.setMode(false)
                }
            }
        }

        findViewById<Button>(R.id.cameraButton).setOnClickListener {
            startCamera()
        }

        findViewById<Button>(R.id.videoButton).setOnClickListener {
            pickVideo()
        }

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                7
            )
        } else {
            startCamera()
        }
    }

    private fun startCamera() {

        videoImage.visibility = View.GONE
        preview.visibility = View.VISIBLE

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            val cameraProvider = cameraProviderFuture.get()

            val cameraPreview =
                Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(preview.surfaceProvider)
                    }

            val imageAnalysis =
                ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .build()

            imageAnalysis.setAnalyzer(executor) { image ->

                try {
                    val bitmap = image.toBitmap()
                    runDetection(bitmap)
                } finally {
                    image.close()
                }
            }

            cameraProvider.unbindAll()

            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                cameraPreview,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    private fun runDetection(bitmap: Bitmap) {

        val result = detector.detect(bitmap)

        val detections = result.first
        val inferenceMs = result.second

        val currentTime = System.nanoTime()

        val elapsed =
            (currentTime - lastFrameTime).coerceAtLeast(1L)

        val fps =
            1_000_000_000.0 / elapsed.toDouble()

        lastFrameTime = currentTime

        val currentCpuTime =
            Process.getElapsedCpuTime()

        val currentWallTime =
            System.currentTimeMillis()

        val wallDifference =
            (currentWallTime - previousWallTime)
                .coerceAtLeast(1L)

        val cpuDifference =
            currentCpuTime - previousCpuTime

        val cpuUsage =
            (
                cpuDifference.toDouble() /
                    wallDifference.toDouble() *
                    100.0
            ).coerceIn(0.0, 100.0)

        previousCpuTime = currentCpuTime
        previousWallTime = currentWallTime

        runOnUiThread {

            overlay.setDetections(
                detections,
                640,
                640
            )

            monitor.text =
                """
                MODE: ${detector.mode}
                NPU: $acceleratorStatus
                CPU: %.1f%%
                FPS: %.1f
                INFERENCE: %.1f ms
                OBJECTS: %d
                """.trimIndent().format(
                    cpuUsage,
                    fps,
                    inferenceMs,
                    detections.size
                )
        }
    }

    private fun pickVideo() {

        val intent =
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "video/*"
                addCategory(Intent.CATEGORY_OPENABLE)
            }

        startActivityForResult(intent, 9)
    }

    @Deprecated("Deprecated in Java")
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
            requestCode != 9 ||
            resultCode != RESULT_OK
        ) {
            return
        }

        val videoUri = data?.data ?: return

        preview.visibility = View.GONE
        videoImage.visibility = View.VISIBLE

        executor.execute {

            val retriever =
                MediaMetadataRetriever()

            try {

                retriever.setDataSource(
                    this,
                    videoUri
                )

                val duration =
                    retriever.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_DURATION
                    )?.toLongOrNull() ?: 0L

                var timeMs = 0L

                while (timeMs < duration) {

                    val bitmap =
                        retriever.getFrameAtTime(
                            timeMs * 1000L,
                            MediaMetadataRetriever.OPTION_CLOSEST
                        )

                    if (bitmap == null) {
                        timeMs += 100L
                        continue
                    }

                    runOnUiThread {
                        videoImage.setImageBitmap(bitmap)
                    }

                    runDetection(bitmap)

                    timeMs += 100L
                }

            } catch (e: Exception) {

                runOnUiThread {
                    monitor.text =
                        "VIDEO ERROR: ${e.message}"
                }

            } finally {

                retriever.release()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
    }
}
