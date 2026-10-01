package com.thesis.t2520738

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class YoloDetector(private val ctx: Context) {

    companion object {
        val NAMES = arrayOf(
            "bicycle",
            "bus",
            "car",
            "cng",
            "cones",
            "dog",
            "leguna",
            "motorcycle",
            "mpv",
            "pedestrian",
            "pickup",
            "rickshaw",
            "roadDivider",
            "trafficLight",
            "trafficSign",
            "trees",
            "truck",
            "van",
            "wheelBarrow"
        )

        private const val INPUT_SIZE = 640
        private const val NUM_CLASSES = 19
        private const val NUM_PREDICTIONS = 8400
        private const val CONFIDENCE_THRESHOLD = 0.35f
        private const val NMS_THRESHOLD = 0.45f
    }

    private val env = OrtEnvironment.getEnvironment()

    private val modelBytes: ByteArray =
        ctx.assets.open("best.onnx").use {
            it.readBytes()
        }

    private var session: OrtSession? = null

    var mode = "CPU"
        private set

    init {
        setMode(false)
    }

    @Synchronized
    fun setMode(useNpu: Boolean): String {

        /*
         * IMPORTANT:
         *
         * We create the NEW session first.
         * We do NOT close the current working session until
         * the new session has successfully initialized.
         *
         * Therefore, if NNAPI initialization fails,
         * CPU mode continues working.
         */

        val options = OrtSession.SessionOptions()

        try {

            if (useNpu) {

                /*
                 * Enable Android NNAPI.
                 *
                 * CPU_DISABLED is intentionally NOT used.
                 *
                 * NNAPI may execute supported operations using
                 * available Android neural acceleration hardware.
                 *
                 * Unsupported operations are allowed to fall
                 * back instead of crashing/failing the session.
                 */
                options.addNnapi()
            }

            val newSession =
                env.createSession(
                    modelBytes,
                    options
                )

            val oldSession = session

            session = newSession

            mode =
                if (useNpu) {
                    "NNAPI"
                } else {
                    "CPU"
                }

            oldSession?.close()

            return if (useNpu) {
                "NNAPI ACTIVE"
            } else {
                "OFF"
            }

        } catch (e: Exception) {

            /*
             * If NNAPI cannot initialize, keep the existing
             * working session instead of crashing the app.
             */

            if (session == null) {

                val cpuOptions =
                    OrtSession.SessionOptions()

                try {

                    session =
                        env.createSession(
                            modelBytes,
                            cpuOptions
                        )

                    mode = "CPU"

                } finally {
                    cpuOptions.close()
                }
            }

            throw RuntimeException(
                "NNAPI initialization failed: ${e.message}",
                e
            )

        } finally {
            options.close()
        }
    }

    @Synchronized
    fun detect(
        src: Bitmap
    ): Pair<List<Detection>, Double> {

        if (session == null) {
            setMode(false)
        }

        val currentSession =
            session ?: throw IllegalStateException(
                "ONNX Runtime session unavailable"
            )

        val resized =
            Bitmap.createScaledBitmap(
                src,
                INPUT_SIZE,
                INPUT_SIZE,
                true
            )

        val pixels =
            IntArray(
                INPUT_SIZE * INPUT_SIZE
            )

        resized.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        val plane =
            INPUT_SIZE * INPUT_SIZE

        val inputData =
            FloatArray(
                3 * plane
            )

        for (i in pixels.indices) {

            val pixel = pixels[i]

            inputData[i] =
                ((pixel shr 16) and 255) / 255.0f

            inputData[plane + i] =
                ((pixel shr 8) and 255) / 255.0f

            inputData[(2 * plane) + i] =
                (pixel and 255) / 255.0f
        }

        val tensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputData),
                longArrayOf(
                    1,
                    3,
                    INPUT_SIZE.toLong(),
                    INPUT_SIZE.toLong()
                )
            )

        try {

            val inputName =
                currentSession.inputNames.first()

            val startTime =
                System.nanoTime()

            val results =
                currentSession.run(
                    mapOf(
                        inputName to tensor
                    )
                )

            try {

                val inferenceMs =
                    (System.nanoTime() - startTime) /
                        1_000_000.0

                @Suppress("UNCHECKED_CAST")
                val output =
                    results[0].value
                            as Array<Array<FloatArray>>

                val predictions =
                    output[0]

                val candidates =
                    ArrayList<Detection>()

                for (
                    predictionIndex
                    in 0 until NUM_PREDICTIONS
                ) {

                    var bestScore = 0.0f
                    var bestClass = 0

                    for (
                        classIndex
                        in 0 until NUM_CLASSES
                    ) {

                        val score =
                            predictions[
                                4 + classIndex
                            ][predictionIndex]

                        if (score > bestScore) {

                            bestScore = score
                            bestClass = classIndex
                        }
                    }

                    if (
                        bestScore >=
                        CONFIDENCE_THRESHOLD
                    ) {

                        val cx =
                            predictions[0][predictionIndex]

                        val cy =
                            predictions[1][predictionIndex]

                        val width =
                            predictions[2][predictionIndex]

                        val height =
                            predictions[3][predictionIndex]

                        candidates.add(
                            Detection(
                                cx - width / 2.0f,
                                cy - height / 2.0f,
                                cx + width / 2.0f,
                                cy + height / 2.0f,
                                bestScore,
                                bestClass
                            )
                        )
                    }
                }

                val detections =
                    nms(
                        candidates,
                        NMS_THRESHOLD
                    )

                return Pair(
                    detections,
                    inferenceMs
                )

            } finally {
                results.close()
            }

        } finally {
            tensor.close()
        }
    }

    private fun nms(
        detections: List<Detection>,
        threshold: Float
    ): List<Detection> {

        val output =
            ArrayList<Detection>()

        for (
            classId
            in 0 until NUM_CLASSES
        ) {

            val remaining =
                detections
                    .filter {
                        it.classId == classId
                    }
                    .sortedByDescending {
                        it.score
                    }
                    .toMutableList()

            while (
                remaining.isNotEmpty()
            ) {

                val best =
                    remaining.removeAt(0)

                output.add(best)

                remaining.removeAll {
                    iou(best, it) > threshold
                }
            }
        }

        return output
    }

    private fun iou(
        a: Detection,
        b: Detection
    ): Float {

        val x1 =
            max(a.x1, b.x1)

        val y1 =
            max(a.y1, b.y1)

        val x2 =
            min(a.x2, b.x2)

        val y2 =
            min(a.y2, b.y2)

        val intersection =
            max(0.0f, x2 - x1) *
                max(0.0f, y2 - y1)

        val areaA =
            max(0.0f, a.x2 - a.x1) *
                max(0.0f, a.y2 - a.y1)

        val areaB =
            max(0.0f, b.x2 - b.x1) *
                max(0.0f, b.y2 - b.y1)

        return intersection /
            (
                areaA +
                    areaB -
                    intersection +
                    0.000001f
            )
    }
}
