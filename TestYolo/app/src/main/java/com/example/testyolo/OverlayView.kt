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

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.GREEN
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
        invalidate()
    }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        if (imgW == 0 || imgH == 0) return

        val vw = width.toFloat();
        val vh = height.toFloat()
        val (sw, sh) = if (rotation % 180 == 0) imgW.toFloat() to imgH.toFloat()
        else imgH.toFloat() to imgW.toFloat()
        val scale = max(vw / sw, vh / sh)
        val dx = (vw - sw * scale) / 2f
        val dy = (vh - sh * scale) / 2f

        c.save()
        when (rotation) {
            90 -> {
                c.translate(vw, 0f); c.rotate(90f)
            }

            180 -> {
                c.translate(vw, vh); c.rotate(180f)
            }

            270 -> {
                c.translate(0f, vh); c.rotate(270f)
            }
        }
        c.translate(dx, dy)
        c.scale(scale, scale)

        val pad = 6f
        for (d in dets) {
            val x1 = d[0];
            val y1 = d[1];
            val x2 = d[2];
            val y2 = d[3]
            val score = d[4]
            val clsIdx = d[5].toInt()

            // бокс
            c.drawRect(x1, y1, x2, y2, boxPaint)

            // подпись
            val name = labels.getOrNull(clsIdx) ?: clsIdx.toString()
            val pct = (score * 100f).coerceIn(0f, 100f)
            val label = "$name ${"%.1f".format(Locale.US, pct)}%"

            val tw = textPaint.measureText(label)
// точнее, чем textSize:
            val th = textPaint.fontMetrics.run { bottom - top }

            var tx = x1 + 2f
// не выезжать за правую грань бокса
            if (tx + tw + pad > x2) {
                tx = max(x1 + 2f, x2 - tw - pad)
            }

            val drawAbove = (y1 - th - 2 * pad) >= 0f
            val bgLeft = tx - pad
            val bgTop = if (drawAbove) y1 - th - 2 * pad else y1
            val bgRight = tx + tw + pad
            val bgBottom = if (drawAbove) y1 else y1 + th + 2 * pad
            val textY = if (drawAbove) (y1 - pad) else (y1 + th + pad)

            c.drawRect(bgLeft, bgTop, bgRight, bgBottom, bgPaint)
            c.drawText(label, tx, textY, textPaint)
        }
    }
}
