package com.example.testyolo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import org.json.JSONObject
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

class CliBenchActivity : ComponentActivity() {

    companion object {
        private const val TAG = "CliBench"
        private const val RESULT_TAG_DEFAULT = "XTRIM_RESULT"
        private const val MAX_ASSET_IMAGES_DEFAULT = 256
    }

    object YoloBridge {
        init {
            System.loadLibrary("ncnn")
            System.loadLibrary("yolo")
        }

        external fun init(assetMgr: android.content.res.AssetManager): Boolean

        // NEW: загрузка модели из ФАЙЛОВОЙ СИСТЕМЫ (adb push)
        external fun loadFromFile(paramPath: String, binPath: String, inputSize: Int, numThreads: Int): Boolean

        // детект с заданным inputSize (imgsz)
        external fun detectRgbaWithSize(
            rgba: ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int,
            conf: Float, iou: Float, inputSize: Int
        ): Array<FloatArray>

        external fun release()

        external fun setOptimized(enabled: Boolean)
        external fun isOptimized(): Boolean
    }

    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Берём параметры из Intent extras (их даёт ПК-пайплайн через adb am start ...)
        val paramPath = intent.getStringExtra("param")
        val binPath = intent.getStringExtra("bin")
        val dataset = intent.getStringExtra("dataset") ?: "cifar"
        val runId = intent.getStringExtra("run_id") ?: ""
        val resultTag = intent.getStringExtra("result_tag") ?: RESULT_TAG_DEFAULT

        val imgsz = intent.getIntExtra("imgsz", 640)
        val loops = intent.getIntExtra("loops", 50).coerceAtLeast(1)
        val warmup = intent.getIntExtra("warmup", 10).coerceAtLeast(0)
        val threads = intent.getIntExtra("threads", 4).coerceAtLeast(1)
        val conf = intent.getFloatExtra("conf", 0.25f)
        val iou = intent.getFloatExtra("iou", 0.45f)
        val optimized = intent.getBooleanExtra("optimized", true)

        val maxAssetImages = intent.getIntExtra("max_images", MAX_ASSET_IMAGES_DEFAULT).coerceAtLeast(1)

        executor.execute {
            val json = try {
                benchOnce(
                    paramPath = paramPath,
                    binPath = binPath,
                    dataset = dataset,
                    runId = runId,
                    imgsz = imgsz,
                    loops = loops,
                    warmup = warmup,
                    threads = threads,
                    conf = conf,
                    iou = iou,
                    optimized = optimized,
                    maxAssetImages = maxAssetImages
                )
            } catch (e: Exception) {
                JSONObject().apply {
                    put("ok", false)
                    put("run_id", runId)
                    put("error", e.toString())
                }
            }

            // ВАЖНО: ровно сюда смотрит ПК (logcat -s XTRIM_RESULT:I)
            Log.i(resultTag, json.toString())

            // быстро закрываем activity
            try {
                YoloBridge.release()
            } catch (_: Exception) {}

            runOnUiThread {
                finish()
            }
        }
    }

    private data class ImageData(val buffer: ByteBuffer, val width: Int, val height: Int)

    private fun benchOnce(
        paramPath: String?,
        binPath: String?,
        dataset: String,
        runId: String,
        imgsz: Int,
        loops: Int,
        warmup: Int,
        threads: Int,
        conf: Float,
        iou: Float,
        optimized: Boolean,
        maxAssetImages: Int
    ): JSONObject {

        if (paramPath.isNullOrBlank() || binPath.isNullOrBlank()) {
            throw IllegalArgumentException("Missing extras: param/bin")
        }

        // 1) Инициализация native
        val okInit = YoloBridge.init(assets)
        if (!okInit) {
            throw RuntimeException("YoloBridge.init() failed")
        }

        // 2) Режим оптимизаций (FP16/Winograd/SGEMM/packing) — управляет native
        YoloBridge.setOptimized(optimized)

        // 3) Грузим модель из путей (adb push)
        val okLoad = YoloBridge.loadFromFile(paramPath, binPath, imgsz, threads)
        if (!okLoad) {
            throw RuntimeException("loadFromFile failed: param=$paramPath bin=$binPath imgsz=$imgsz threads=$threads")
        }

        // 4) Загружаем изображения из assets/<dataset>/
        val imageList = loadImagesFromAssets(dataset, maxAssetImages)
        if (imageList.isEmpty()) {
            throw RuntimeException("No images in assets/$dataset/")
        }

        // 5) Warmup
        val warmImg = imageList[0]
        for (i in 0 until warmup) {
            warmImg.buffer.rewind()
            YoloBridge.detectRgbaWithSize(
                warmImg.buffer,
                warmImg.width, warmImg.height,
                warmImg.width * 4,
                0,
                conf, iou,
                imgsz
            )
        }

        // 6) Тайминг: loops = число inference (как у benchncnn)
        val times = DoubleArray(loops)
        var detSum = 0L

        for (i in 0 until loops) {
            val img = imageList[i % imageList.size]
            img.buffer.rewind()

            val t0 = SystemClock.elapsedRealtimeNanos()
            val dets = YoloBridge.detectRgbaWithSize(
                img.buffer,
                img.width, img.height,
                img.width * 4,
                0,
                conf, iou,
                imgsz
            )
            val t1 = SystemClock.elapsedRealtimeNanos()

            times[i] = (t1 - t0) / 1_000_000.0
            detSum += dets.size.toLong()
        }

        val avg = times.average()
        val min = times.minOrNull() ?: avg
        val max = times.maxOrNull() ?: avg

        var varSum = 0.0
        for (t in times) {
            val d = t - avg
            varSum += d * d
        }
        val std = kotlin.math.sqrt(varSum / times.size.coerceAtLeast(1))
        val detAvg = detSum.toDouble() / times.size.toDouble()

        return JSONObject().apply {
            put("ok", true)
            put("run_id", runId)
            put("avg_ms", avg)
            put("min_ms", min)
            put("max_ms", max)
            put("std_ms", std)
            put("n", times.size)
            put("images", imageList.size)
            put("loops", loops)
            put("warmup", warmup)
            put("imgsz", imgsz)
            put("threads", threads)
            put("optimized", optimized)
            put("det_avg", detAvg)
            put("dataset", dataset)
            put("param", paramPath)
            put("bin", binPath)
        }
    }

    private fun loadImagesFromAssets(dataset: String, maxImages: Int): List<ImageData> {
        val names = try {
            assets.list(dataset)
                ?.filter { it.endsWith(".png", true) || it.endsWith(".jpg", true) || it.endsWith(".jpeg", true) }
                ?.sorted()
                ?.take(maxImages)
                ?: emptyList()
        } catch (e: IOException) {
            emptyList()
        }

        val out = ArrayList<ImageData>(names.size)
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }

        for (fn in names) {
            try {
                assets.open("$dataset/$fn").use { stream ->
                    val bmp = BitmapFactory.decodeStream(stream, null, opts) ?: return@use
                    val buf = bitmapToRgbaBuffer(bmp)
                    out.add(ImageData(buf, bmp.width, bmp.height))
                    bmp.recycle()
                }
            } catch (_: Exception) {}
        }
        return out
    }

    private fun bitmapToRgbaBuffer(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val buffer = ByteBuffer.allocateDirect(width * height * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixel in pixels) {
            buffer.put(((pixel shr 16) and 0xFF).toByte()) // R
            buffer.put(((pixel shr 8) and 0xFF).toByte())  // G
            buffer.put((pixel and 0xFF).toByte())          // B
            buffer.put(((pixel shr 24) and 0xFF).toByte()) // A
        }
        buffer.rewind()
        return buffer
    }
}