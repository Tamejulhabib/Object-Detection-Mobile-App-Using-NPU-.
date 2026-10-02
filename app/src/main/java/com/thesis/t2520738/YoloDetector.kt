package com.thesis.t2520738

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    private val context: Context
) {

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

        private const val CONFIDENCE_THRESHOLD =
            0.35f

        private const val NMS_THRESHOLD =
            0.45f
    }

    private val env =
        OrtEnvironment.getEnvironment()

    private val modelBytes =
        context.assets
            .open("best.onnx")
            .use {
                it.readBytes()
            }

    private var session:
        OrtSession? = null

    var mode = "CPU"
        private set

    init {

        setMode(false)
    }

    /*
     * ==============================================================
     * CPU / NNAPI SESSION
     * ==============================================================
     */

    @Synchronized
    fun setMode(
        useNpu: Boolean
    ): String {

        val options =
            OrtSession.SessionOptions()

        try {

            if (useNpu) {

                /*
                 * IMPORTANT:
                 *
                 * No CPU_DISABLED.
                 *
                 * No FP16 during accuracy debugging.
                 *
                 * NNAPI is only enabled after MainActivity
                 * has validated the supported experimental
                 * device.
                 */

                options.addNnapi()
            }

            /*
             * Create new session before destroying
             * the currently working session.
             */

            val newSession =
                env.createSession(
                    modelBytes,
                    options
                )

            val oldSession =
                session

            session =
                newSession

            mode =
                if (useNpu) {
                    "NNAPI"
                } else {
                    "CPU"
                }

            oldSession?.close()

            return if (useNpu) {
                "ACTIVE"
            } else {
                "OFF"
            }

        } catch (e: Exception) {

            /*
             * If NNAPI initialization fails,
             * preserve/create CPU execution.
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

    /*
     * ==============================================================
     * DETECTION
     * ==============================================================
     */

    @Synchronized
    fun detect(
        source: Bitmap
    ): List<Detection> {

        if (
            session == null
        ) {

            setMode(false)
        }

        val currentSession =
            session
                ?: return emptyList()

        val originalWidth =
            source.width

        val originalHeight =
            source.height

        if (
            originalWidth <= 0 ||
            originalHeight <= 0
        ) {

            return emptyList()
        }

        /*
         * ==========================================================
         * LETTERBOX
         * ==========================================================
         *
         * Do NOT stretch a rectangular image
         * directly into 640x640.
         */

        val scale =
            min(
                INPUT_SIZE.toFloat() /
                    originalWidth.toFloat(),

                INPUT_SIZE.toFloat() /
                    originalHeight.toFloat()
            )

        val scaledWidth =
            (originalWidth * scale)
                .toInt()
                .coerceAtLeast(1)

        val scaledHeight =
            (originalHeight * scale)
                .toInt()
                .coerceAtLeast(1)

        val resized =
            Bitmap.createScaledBitmap(
                source,
                scaledWidth,
                scaledHeight,
                true
            )

        val letterboxed =
            Bitmap.createBitmap(
                INPUT_SIZE,
                INPUT_SIZE,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(letterboxed)

        /*
         * Standard YOLO-style padding value.
         */

        canvas.drawColor(
            Color.rgb(
                114,
                114,
                114
            )
        )

        val padX =
            (
                INPUT_SIZE -
                    scaledWidth
                ) / 2f

        val padY =
            (
                INPUT_SIZE -
                    scaledHeight
                ) / 2f

        val bitmapPaint =
            Paint(
                Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            resized,
            padX,
            padY,
            bitmapPaint
        )

        /*
         * ==========================================================
         * BITMAP -> RGB FLOAT32
         * ==========================================================
         */

        val pixels =
            IntArray(
                INPUT_SIZE *
                    INPUT_SIZE
            )

        letterboxed.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        val plane =
            INPUT_SIZE *
                INPUT_SIZE

        val inputData =
            FloatArray(
                3 * plane
            )

        for (
            i in pixels.indices
        ) {

            val pixel =
                pixels[i]

            /*
             * R
             */

            inputData[i] =
                (
                    (pixel shr 16)
                        and 255
                    ) / 255.0f

            /*
             * G
             */

            inputData[
                plane + i
            ] =
                (
                    (pixel shr 8)
                        and 255
                    ) / 255.0f

            /*
             * B
             */

            inputData[
                (2 * plane) + i
            ] =
                (
                    pixel and 255
                    ) / 255.0f
        }

        val tensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(
                    inputData
                ),
                longArrayOf(
                    1,
                    3,
                    INPUT_SIZE.toLong(),
                    INPUT_SIZE.toLong()
                )
            )

        try {

            val inputName =
                currentSession
                    .inputNames
                    .first()

            val results =
                currentSession.run(
                    mapOf(
                        inputName to tensor
                    )
                )

            try {

                /*
                 * Model:
                 *
                 * output0
                 * [1,23,8400]
                 *
                 * 4 coordinates
                 * 19 class scores
                 */

                @Suppress(
                    "UNCHECKED_CAST"
                )
                val output =
                    results[0].value
                            as Array<
                            Array<FloatArray>
                            >

                val predictions =
                    output[0]

                val candidates =
                    ArrayList<Detection>()

                /*
                 * ==================================================
                 * DECODE YOLO OUTPUT
                 * ==================================================
                 */

                for (
                    predictionIndex
                    in 0 until
                        NUM_PREDICTIONS
                ) {

                    var bestScore =
                        0f

                    var bestClass =
                        0

                    for (
                        classIndex
                        in 0 until
                            NUM_CLASSES
                    ) {

                        val score =
                            predictions[
                                4 +
                                    classIndex
                            ][
                                predictionIndex
                            ]

                        if (
                            score >
                            bestScore
                        ) {

                            bestScore =
                                score

                            bestClass =
                                classIndex
                        }
                    }

                    if (
                        bestScore <
                        CONFIDENCE_THRESHOLD
                    ) {

                        continue
                    }

                    val cx =
                        predictions[0][
                            predictionIndex
                        ]

                    val cy =
                        predictions[1][
                            predictionIndex
                        ]

                    val boxWidth =
                        predictions[2][
                            predictionIndex
                        ]

                    val boxHeight =
                        predictions[3][
                            predictionIndex
                        ]

                    /*
                     * Model coordinates are currently
                     * relative to the 640x640 letterbox.
                     */

                    var x1 =
                        cx -
                            boxWidth / 2f

                    var y1 =
                        cy -
                            boxHeight / 2f

                    var x2 =
                        cx +
                            boxWidth / 2f

                    var y2 =
                        cy +
                            boxHeight / 2f

                    /*
                     * ==================================================
                     * REMOVE LETTERBOX PADDING
                     * ==================================================
                     */

                    x1 =
                        (
                            x1 -
                                padX
                            ) / scale

                    y1 =
                        (
                            y1 -
                                padY
                            ) / scale

                    x2 =
                        (
                            x2 -
                                padX
                            ) / scale

                    y2 =
                        (
                            y2 -
                                padY
                            ) / scale

                    /*
                     * Clamp to actual source image.
                     */

                    x1 =
                        x1.coerceIn(
                            0f,
                            originalWidth.toFloat()
                        )

                    y1 =
                        y1.coerceIn(
                            0f,
                            originalHeight.toFloat()
                        )

                    x2 =
                        x2.coerceIn(
                            0f,
                            originalWidth.toFloat()
                        )

                    y2 =
                        y2.coerceIn(
                            0f,
                            originalHeight.toFloat()
                        )

                    if (
                        x2 <= x1 ||
                        y2 <= y1
                    ) {

                        continue
                    }

                    candidates.add(
                        Detection(
                            x1,
                            y1,
                            x2,
                            y2,
                            bestScore,
                            bestClass
                        )
                    )
                }

                /*
                 * ==================================================
                 * NMS
                 * ==================================================
                 */

                return nms(
                    candidates,
                    NMS_THRESHOLD
                )

            } finally {

                results.close()
            }

        } finally {

            tensor.close()

            if (
                resized !== source
            ) {

                resized.recycle()
            }

            letterboxed.recycle()
        }
    }

    /*
     * ==============================================================
     * NON-MAXIMUM SUPPRESSION
     * ==============================================================
     */

    private fun nms(
        detections: List<Detection>,
        threshold: Float
    ): List<Detection> {

        val output =
            ArrayList<Detection>()

        for (
            classId
            in 0 until
                NUM_CLASSES
        ) {

            val remaining =
                detections
                    .filter {

                        it.classId ==
                            classId
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

                output.add(
                    best
                )

                remaining.removeAll {

                    iou(
                        best,
                        it
                    ) >
                        threshold
                }
            }
        }

        return output
    }

    /*
     * ==============================================================
     * INTERSECTION OVER UNION
     * ==============================================================
     */

    private fun iou(
        a: Detection,
        b: Detection
    ): Float {

        val x1 =
            max(
                a.x1,
                b.x1
            )

        val y1 =
            max(
                a.y1,
                b.y1
            )

        val x2 =
            min(
                a.x2,
                b.x2
            )

        val y2 =
            min(
                a.y2,
                b.y2
            )

        val intersection =
            max(
                0f,
                x2 - x1
            ) *
                max(
                    0f,
                    y2 - y1
                )

        val areaA =
            max(
                0f,
                a.x2 - a.x1
            ) *
                max(
                    0f,
                    a.y2 - a.y1
                )

        val areaB =
            max(
                0f,
                b.x2 - b.x1
            ) *
                max(
                    0f,
                    b.y2 - b.y1
                )

        return intersection /
            (
                areaA +
                    areaB -
                    intersection +
                    0.000001f
                )
    }
}
