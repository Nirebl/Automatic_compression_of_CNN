package com.example.testyolo

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import java.util.Locale
import kotlin.math.max

class OverlayView(context: Context, attrs: AttributeSet): View(context, attrs) {
    private var dets: List<FloatArray> = emptyList()
    private var imgW = 0;
    private var imgH = 0
    private var rotation = 0
    private var labels: List<String> = emptyList()
    private var isSegMode = false  // Flag for segmentation mode

    // Colors for different classes in segmentation mode
    private val segColors = listOf(
        Color.argb(100, 255, 0, 0),    // Red
        Color.argb(100, 0, 255, 0),    // Green
        Color.argb(100, 0, 0, 255),    // Blue
        Color.argb(100, 255, 255, 0),  // Yellow
        Color.argb(100, 255, 0, 255),  // Magenta
        Color.argb(100, 0, 255, 255),  // Cyan
        Color.argb(100, 255, 128, 0),  // Orange
        Color.argb(100, 128, 0, 255),  // Purple
        Color.argb(100, 0, 255, 128),  // Teal
        Color.argb(100, 255, 0, 128),  // Pink
    )

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.GREEN
        isAntiAlias = true
    }
    private val segFillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
    }
    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(160, 0, 0, 0) // полупрозрачный фон под подписью
        isAntiAlias = true
    }

    fun setLabels(labels: List<String>) {
        this.labels = labels
    }

    fun update(imgW: Int, imgH: Int, dets: List<FloatArray>, rotationDeg: Int) {
        this.imgW = imgW; this.imgH = imgH; this.dets = dets; this.rotation = rotationDeg
        this.isSegMode = false
        invalidate()
    }

    fun updateSeg(imgW: Int, imgH: Int, dets: List<FloatArray>, rotationDeg: Int) {
        this.imgW = imgW; this.imgH = imgH; this.dets = dets; this.rotation = rotationDeg
        this.isSegMode = true
        invalidate()
    }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        if (imgW == 0 || imgH == 0) return

        val vw = width.toFloat()
        val vh = height.toFloat()

        // Displayed dimensions after rotation
        val (dispW, dispH) = if (rotation % 180 == 0) imgW.toFloat() to imgH.toFloat()
                             else imgH.toFloat() to imgW.toFloat()

        // Scale to fill the view (matching PreviewView.FILL_CENTER)
        val scale = max(vw / dispW, vh / dispH)
        val offsetX = (vw - dispW * scale) / 2f
        val offsetY = (vh - dispH * scale) / 2f

        val pad = 6f
        for (d in dets) {
            // Original coords from C++ (in camera frame space)
            val ox1 = d[0]; val oy1 = d[1]
            val ox2 = d[2]; val oy2 = d[3]
            val score = d[4]
            val clsIdx = d[5].toInt()

            // Transform original coords to rotated display coords
            // C++ read_pixel_rotated for rot=90: sx=y, sy=srcW-1-x
            // So inverse: original (ox, oy) → rotated (oy, imgW - 1 - ox)
            val (rx1, ry1, rx2, ry2) = when (rotation) {
                90 -> floatArrayOf(
                    oy1, imgW - 1 - ox2,  // top-left in rotated
                    oy2, imgW - 1 - ox1   // bottom-right in rotated
                )
                180 -> floatArrayOf(
                    imgW - 1 - ox2, imgH - 1 - oy2,
                    imgW - 1 - ox1, imgH - 1 - oy1
                )
                270 -> floatArrayOf(
                    imgH - 1 - oy2, ox1,
                    imgH - 1 - oy1, ox2
                )
                else -> floatArrayOf(ox1, oy1, ox2, oy2)  // rotation 0
            }

            // Apply scale and offset to get view coordinates
            val vx1 = rx1 * scale + offsetX
            val vy1 = ry1 * scale + offsetY
            val vx2 = rx2 * scale + offsetX
            val vy2 = ry2 * scale + offsetY

            // In segmentation mode, draw filled box with class-based color
            if (isSegMode && clsIdx >= 0) {
                val colorIdx = ((clsIdx % segColors.size) + segColors.size) % segColors.size
                segFillPaint.color = segColors[colorIdx]
                c.drawRect(vx1, vy1, vx2, vy2, segFillPaint)
            }

            // Draw bounding box
            c.drawRect(vx1, vy1, vx2, vy2, boxPaint)

            // Draw label
            val name = labels.getOrNull(clsIdx) ?: clsIdx.toString()
            val pct = (score * 100f).coerceIn(0f, 100f)
            val label = "$name ${"%.1f".format(Locale.US, pct)}%"

            val tw = textPaint.measureText(label)
            val th = textPaint.fontMetrics.run { bottom - top }

            var tx = vx1 + 2f
            // Don't overflow past the right edge of the box
            if (tx + tw + pad > vx2) {
                tx = max(vx1 + 2f, vx2 - tw - pad)
            }

            val drawAbove = (vy1 - th - 2 * pad) >= 0f
            val bgLeft = tx - pad
            val bgTop = if (drawAbove) vy1 - th - 2 * pad else vy1
            val bgRight = tx + tw + pad
            val bgBottom = if (drawAbove) vy1 else vy1 + th + 2 * pad
            val textY = if (drawAbove) (vy1 - pad) else (vy1 + th + pad)

            c.drawRect(bgLeft, bgTop, bgRight, bgBottom, bgPaint)
            c.drawText(label, tx, textY, textPaint)
        }
    }
}
