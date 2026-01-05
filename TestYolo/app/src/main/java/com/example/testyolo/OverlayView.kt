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
    private var hasContours = false  // Flag for contour-based drawing

    // Colors for different classes in segmentation mode (fill)
    private val segFillColors = listOf(
        Color.argb(80, 255, 0, 0),    // Red
        Color.argb(80, 0, 255, 0),    // Green
        Color.argb(80, 0, 0, 255),    // Blue
        Color.argb(80, 255, 255, 0),  // Yellow
        Color.argb(80, 255, 0, 255),  // Magenta
        Color.argb(80, 0, 255, 255),  // Cyan
        Color.argb(80, 255, 128, 0),  // Orange
        Color.argb(80, 128, 0, 255),  // Purple
        Color.argb(80, 0, 255, 128),  // Teal
        Color.argb(80, 255, 0, 128),  // Pink
    )
    
    // Solid colors for polygon stroke
    private val segStrokeColors = listOf(
        Color.rgb(255, 0, 0),    // Red
        Color.rgb(0, 255, 0),    // Green
        Color.rgb(0, 0, 255),    // Blue
        Color.rgb(255, 255, 0),  // Yellow
        Color.rgb(255, 0, 255),  // Magenta
        Color.rgb(0, 255, 255),  // Cyan
        Color.rgb(255, 128, 0),  // Orange
        Color.rgb(128, 0, 255),  // Purple
        Color.rgb(0, 255, 128),  // Teal
        Color.rgb(255, 0, 128),  // Pink
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
    private val segStrokePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
    }
    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(160, 0, 0, 0) // semi-transparent background for labels
        isAntiAlias = true
    }

    fun setLabels(labels: List<String>) {
        this.labels = labels
    }

    fun update(imgW: Int, imgH: Int, dets: List<FloatArray>, rotationDeg: Int) {
        this.imgW = imgW; this.imgH = imgH; this.dets = dets; this.rotation = rotationDeg
        this.isSegMode = false
        this.hasContours = false
        invalidate()
    }

    fun updateSeg(imgW: Int, imgH: Int, dets: List<FloatArray>, rotationDeg: Int) {
        this.imgW = imgW; this.imgH = imgH; this.dets = dets; this.rotation = rotationDeg
        this.isSegMode = true
        this.hasContours = false
        invalidate()
    }
    
    fun updateSegWithContours(imgW: Int, imgH: Int, dets: List<FloatArray>, rotationDeg: Int) {
        this.imgW = imgW; this.imgH = imgH; this.dets = dets; this.rotation = rotationDeg
        this.isSegMode = true
        this.hasContours = true
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
            val (rx1, ry1, rx2, ry2) = when (rotation) {
                90 -> floatArrayOf(
                    oy1, imgW - 1 - ox2,
                    oy2, imgW - 1 - ox1
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
            
            val colorIdx = ((clsIdx % segFillColors.size) + segFillColors.size) % segFillColors.size

            // Handle segmentation mode with contours
            if (isSegMode && hasContours && d.size > 7) {
                val numContourPoints = d[6].toInt()
                if (numContourPoints >= 3) {
                    val path = Path()
                    var firstX = 0f
                    var firstY = 0f
                    
                    for (i in 0 until numContourPoints) {
                        val px = d[7 + i * 2]
                        val py = d[7 + i * 2 + 1]
                        
                        // Transform contour point to view coordinates
                        val (rpx, rpy) = when (rotation) {
                            90 -> py to (imgW - 1 - px)
                            180 -> (imgW - 1 - px) to (imgH - 1 - py)
                            270 -> (imgH - 1 - py) to px
                            else -> px to py
                        }
                        val vpx = rpx * scale + offsetX
                        val vpy = rpy * scale + offsetY
                        
                        if (i == 0) {
                            path.moveTo(vpx, vpy)
                            firstX = vpx
                            firstY = vpy
                        } else {
                            path.lineTo(vpx, vpy)
                        }
                    }
                    path.close()
                    
                    // Draw filled polygon
                    segFillPaint.color = segFillColors[colorIdx]
                    c.drawPath(path, segFillPaint)
                    
                    // Draw polygon outline
                    segStrokePaint.color = segStrokeColors[colorIdx]
                    c.drawPath(path, segStrokePaint)
                } else {
                    // Fallback to filled box if not enough contour points
                    segFillPaint.color = segFillColors[colorIdx]
                    c.drawRect(vx1, vy1, vx2, vy2, segFillPaint)
                    boxPaint.color = segStrokeColors[colorIdx]
                    c.drawRect(vx1, vy1, vx2, vy2, boxPaint)
                    boxPaint.color = Color.GREEN  // Reset
                }
            } else if (isSegMode && clsIdx >= 0) {
                // Old seg mode without contours - draw filled box
                segFillPaint.color = segFillColors[colorIdx]
                c.drawRect(vx1, vy1, vx2, vy2, segFillPaint)
                c.drawRect(vx1, vy1, vx2, vy2, boxPaint)
            } else {
                // Detection mode - just draw bounding box
                c.drawRect(vx1, vy1, vx2, vy2, boxPaint)
            }

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
