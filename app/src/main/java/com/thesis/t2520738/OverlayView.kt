package com.thesis.t2520738

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

class OverlayView(
    context: Context,
    attrs: AttributeSet?
) : View(
    context,
    attrs
) {

    private var detections:
        List<Detection> =
        emptyList()

    private var imageWidth =
        1

    private var imageHeight =
        1

    private val boxPaint =
        Paint().apply {

            style =
                Paint.Style.STROKE

            strokeWidth =
                4f

            color =
                Color.GREEN

            isAntiAlias =
                true
        }

    private val textPaint =
        Paint().apply {

            style =
                Paint.Style.FILL

            textSize =
                30f

            color =
                Color.WHITE

            isAntiAlias =
                true
        }

    private val textBackgroundPaint =
        Paint().apply {

            style =
                Paint.Style.FILL

            color =
                Color.argb(
                    180,
                    0,
                    0,
                    0
                )
        }

    /*
     * ==============================================================
     * SET DETECTIONS
     * ==============================================================
     */

    fun setDetections(
        newDetections:
            List<Detection>,
        sourceWidth: Int,
        sourceHeight: Int
    ) {

        detections =
            newDetections

        imageWidth =
            sourceWidth
                .coerceAtLeast(1)

        imageHeight =
            sourceHeight
                .coerceAtLeast(1)

        postInvalidate()
    }

    /*
     * ==============================================================
     * CLEAR
     * ==============================================================
     */

    fun clear() {

        detections =
            emptyList()

        postInvalidate()
    }

    /*
     * ==============================================================
     * DRAW
     * ==============================================================
     */

    override fun onDraw(
        canvas: Canvas
    ) {

        super.onDraw(
            canvas
        )

        if (
            width <= 0 ||
            height <= 0 ||
            imageWidth <= 0 ||
            imageHeight <= 0
        ) {

            return
        }

        /*
         * FIT_CENTER mapping.
         *
         * This matches MainActivity:
         *
         * PreviewView.ScaleType.FIT_CENTER
         * ImageView.ScaleType.FIT_CENTER
         */

        val scale =
            min(
                width.toFloat() /
                    imageWidth.toFloat(),

                height.toFloat() /
                    imageHeight.toFloat()
            )

        val displayedWidth =
            imageWidth *
                scale

        val displayedHeight =
            imageHeight *
                scale

        val offsetX =
            (
                width -
                    displayedWidth
                ) / 2f

        val offsetY =
            (
                height -
                    displayedHeight
                ) / 2f

        /*
         * Draw every detection.
         */

        for (
            detection
            in detections
        ) {

            val left =
                detection.x1 *
                    scale +
                    offsetX

            val top =
                detection.y1 *
                    scale +
                    offsetY

            val right =
                detection.x2 *
                    scale +
                    offsetX

            val bottom =
                detection.y2 *
                    scale +
                    offsetY

            /*
             * Bounding box.
             */

            canvas.drawRect(
                left,
                top,
                right,
                bottom,
                boxPaint
            )

            /*
             * Class name.
             */

            val className =
                if (
                    detection.classId
                    in
                    YoloDetector
                        .NAMES
                        .indices
                ) {

                    YoloDetector
                        .NAMES[
                            detection.classId
                        ]

                } else {

                    "object"
                }

            val confidence =
                (
                    detection.score *
                        100
                    ).toInt()

            val label =
                "$className $confidence%"

            val textWidth =
                textPaint.measureText(
                    label
                )

            val textHeight =
                textPaint.textSize +
                    10f

            var labelTop =
                top -
                    textHeight

            if (
                labelTop <
                0f
            ) {

                labelTop =
                    top
            }

            /*
             * Label background.
             */

            canvas.drawRect(
                left,
                labelTop,
                left +
                    textWidth +
                    12f,
                labelTop +
                    textHeight,
                textBackgroundPaint
            )

            /*
             * Label.
             */

            canvas.drawText(
                label,
                left + 6f,
                labelTop +
                    textPaint.textSize,
                textPaint
            )
        }
    }
}
