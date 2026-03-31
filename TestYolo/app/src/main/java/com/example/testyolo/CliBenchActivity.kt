package com.example.testyolo

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import org.json.JSONObject
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.min
import kotlin.math.roundToInt

class CliBenchActivity : ComponentActivity() {

    companion object {
        private const val TAG = "CliBench"
        private const val RESULT_TAG_DEFAULT = "XTRIM_RESULT"
        private const val MAX_ASSET_IMAGES_DEFAULT = 256
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val ortEnv: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    object YoloBridge {
        init {
            System.loadLibrary("ncnn")
            System.loadLibrary("yolo")
        }

        external fun init(assetMgr: android.content.res.AssetManager): Boolean
        external fun loadFromFile(paramPath: String, binPath: String, inputSize: Int, numThreads: Int): Boolean
        external fun detectRgbaWithSize(
            rgba: ByteBuffer,
            width: Int,
            height: Int,
            rowStride: Int,
            rotationDeg: Int,
            conf: Float,
            iou: Float,
            inputSize: Int
        ): Array<FloatArray>

        external fun release()
        external fun setOptimized(enabled: Boolean)
        external fun isOptimized(): Boolean
    }

    private data class ImageData(val buffer: ByteBuffer, val width: Int, val height: Int)
    private data class OrtInputShape(val n: Long, val c: Long, val h: Long, val w: Long)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val backend = (intent.getStringExtra("backend") ?: "ncnn").trim().lowercase()
        val provider = (intent.getStringExtra("provider") ?: "xnnpack").trim().lowercase()

        val paramPath = intent.getStringExtra("param")
        val binPath = intent.getStringExtra("bin")
        val modelPath = intent.getStringExtra("model")

        val dataset = intent.getStringExtra("dataset") ?: "cifar"
        val runId = intent.getStringExtra("run_id") ?: ""
        val resultTag = intent.getStringExtra("result_tag") ?: RESULT_TAG_DEFAULT

        val imgsz = intent.getIntExtra("imgsz", 640).coerceAtLeast(32)
        val loops = intent.getIntExtra("loops", 50).coerceAtLeast(1)
        val warmup = intent.getIntExtra("warmup", 10).coerceAtLeast(0)
        val threads = intent.getIntExtra("threads", 4).coerceAtLeast(1)
        val conf = intent.getFloatExtra("conf", 0.25f)
        val iou = intent.getFloatExtra("iou", 0.45f)
        val optimized = intent.getBooleanExtra("optimized", true)
        val maxAssetImages = intent.getIntExtra("max_images", MAX_ASSET_IMAGES_DEFAULT).coerceAtLeast(1)

        executor.execute {
            val json = try {
                when (backend) {
                    "ort", "onnx", "ort_android" -> {
                        benchOrtOnce(
                            modelPath = modelPath,
                            provider = provider,
                            dataset = dataset,
                            runId = runId,
                            imgsz = imgsz,
                            loops = loops,
                            warmup = warmup,
                            threads = threads,
                            conf = conf,
                            iou = iou,
                            maxAssetImages = maxAssetImages
                        )
                    }

                    else -> {
                        benchNcnnOnce(
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
                    }
                }
            } catch (e: Exception) {
                JSONObject().apply {
                    put("ok", false)
                    put("run_id", runId)
                    put("backend", backend)
                    put("error", e.toString())
                    put("provider", provider)
                    put("model", modelPath ?: "")
                }
            }

            Log.i(resultTag, json.toString())

            try {
                if (backend != "ort" && backend != "onnx" && backend != "ort_android") {
                    YoloBridge.release()
                }
            } catch (_: Exception) {
            }

            runOnUiThread { finish() }
        }
    }

    private fun benchNcnnOnce(
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

        val okInit = YoloBridge.init(assets)
        if (!okInit) {
            throw RuntimeException("YoloBridge.init() failed")
        }

        YoloBridge.setOptimized(optimized)

        val okLoad = YoloBridge.loadFromFile(paramPath, binPath, imgsz, threads)
        if (!okLoad) {
            throw RuntimeException("loadFromFile failed: param=$paramPath bin=$binPath imgsz=$imgsz threads=$threads")
        }

        val imageList = loadImagesFromAssets(dataset, maxAssetImages)
        if (imageList.isEmpty()) {
            throw RuntimeException("No images in assets/$dataset/")
        }

        val warmImg = imageList[0]
        repeat(warmup) {
            warmImg.buffer.rewind()
            YoloBridge.detectRgbaWithSize(
                warmImg.buffer,
                warmImg.width,
                warmImg.height,
                warmImg.width * 4,
                0,
                conf,
                iou,
                imgsz
            )
        }

        val times = DoubleArray(loops)
        var detSum = 0L

        for (i in 0 until loops) {
            val img = imageList[i % imageList.size]
            img.buffer.rewind()

            val t0 = SystemClock.elapsedRealtimeNanos()
            val dets = YoloBridge.detectRgbaWithSize(
                img.buffer,
                img.width,
                img.height,
                img.width * 4,
                0,
                conf,
                iou,
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
            put("backend", "ncnn")
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

    private fun benchOrtOnce(
        modelPath: String?,
        provider: String,
        dataset: String,
        runId: String,
        imgsz: Int,
        loops: Int,
        warmup: Int,
        threads: Int,
        conf: Float,
        iou: Float,
        maxAssetImages: Int
    ): JSONObject {
        if (modelPath.isNullOrBlank()) {
            throw IllegalArgumentException("Missing extra: model")
        }

        val bitmaps = loadBitmapsFromAssets(dataset, maxAssetImages)
        if (bitmaps.isEmpty()) {
            throw RuntimeException("No images in assets/$dataset/")
        }

        val session = createOrtSession(modelPath, provider, threads)
        try {
            val inputName = session.inputNames.firstOrNull()
                ?: throw RuntimeException("ORT model has no inputs")

            val inputShape = resolveInputShape(session, inputName, imgsz)
            val inputArrayShape = longArrayOf(inputShape.n, inputShape.c, inputShape.h, inputShape.w)

            repeat(warmup) {
                val bmp = bitmaps[it % bitmaps.size]
                val chw = bitmapToFloatCHWLetterbox(bmp, inputShape.w.toInt(), inputShape.h.toInt())
                val tensor = OnnxTensor.createTensor(ortEnv, chw, inputArrayShape)
                try {
                    val result = session.run(mapOf(inputName to tensor))
                    try {
                        // Просто дожидаемся выполнения графа и освобождаем выходы.
                    } finally {
                        result.close()
                    }
                } finally {
                    tensor.close()
                }
            }

            val times = DoubleArray(loops)
            var outputCountSum = 0L

            for (i in 0 until loops) {
                val bmp = bitmaps[i % bitmaps.size]

                val t0 = SystemClock.elapsedRealtimeNanos()

                val chw = bitmapToFloatCHWLetterbox(bmp, inputShape.w.toInt(), inputShape.h.toInt())
                val tensor = OnnxTensor.createTensor(ortEnv, chw, inputArrayShape)

                try {
                    val result = session.run(mapOf(inputName to tensor))
                    try {
                        outputCountSum += result.size().toLong()
                    } finally {
                        result.close()
                    }
                } finally {
                    tensor.close()
                }

                val t1 = SystemClock.elapsedRealtimeNanos()
                times[i] = (t1 - t0) / 1_000_000.0
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
            val outAvg = outputCountSum.toDouble() / times.size.toDouble()

            return JSONObject().apply {
                put("ok", true)
                put("backend", "ort")
                put("provider", provider)
                put("run_id", runId)
                put("avg_ms", avg)
                put("min_ms", min)
                put("max_ms", max)
                put("std_ms", std)
                put("n", times.size)
                put("images", bitmaps.size)
                put("loops", loops)
                put("warmup", warmup)
                put("imgsz", imgsz)
                put("threads", threads)
                put("conf", conf.toDouble())
                put("iou", iou.toDouble())
                put("model", modelPath)
                put("input_name", inputName)
                put("input_h", inputShape.h)
                put("input_w", inputShape.w)
                put("output_count_avg", outAvg)
                put("dataset", dataset)
            }
        } finally {
            try {
                session.close()
            } catch (_: Exception) {
            }
        }
    }

    private fun createOrtSession(modelPath: String, provider: String, threads: Int): OrtSession {
        val so = OrtSession.SessionOptions()
        try {
            so.setIntraOpNumThreads(threads)
            so.setInterOpNumThreads(1)
            so.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            so.setMemoryPatternOptimization(true)
            so.setCPUArenaAllocator(true)
            so.setSessionLogVerbosityLevel(0)

            when (provider.lowercase()) {
                "xnnpack" -> {
                    try {
                        so.addXnnpack(
                            mapOf(
                                "intra_op_num_threads" to threads.toString()
                            )
                        )
                        Log.i(TAG, "ORT provider: XNNPACK")
                    } catch (e: Throwable) {
                        Log.w(TAG, "XNNPACK unavailable, fallback to CPU: $e")
                    }
                }

                "nnapi" -> {
                    try {
                        so.addNnapi(0)
                        Log.i(TAG, "ORT provider: NNAPI")
                    } catch (e: Throwable) {
                        Log.w(TAG, "NNAPI unavailable, fallback to CPU: $e")
                    }
                }

                else -> {
                    Log.i(TAG, "ORT provider: CPU default")
                }
            }

            return ortEnv.createSession(modelPath, so)
        } finally {
            so.close()
        }
    }

    private fun resolveInputShape(session: OrtSession, inputName: String, fallbackImgsz: Int): OrtInputShape {
        val nodeInfo: NodeInfo = session.inputInfo[inputName]
            ?: throw RuntimeException("Input info missing for $inputName")

        val tensorInfo = nodeInfo.info as? TensorInfo
            ?: throw RuntimeException("Input $inputName is not a tensor")

        val shape = tensorInfo.shape
        if (shape.size != 4) {
            throw RuntimeException("Expected 4D input, got shape=${shape.contentToString()}")
        }

        val n = if (shape[0] > 0) shape[0] else 1L
        val c = if (shape[1] > 0) shape[1] else 3L
        val h = if (shape[2] > 0) shape[2] else fallbackImgsz.toLong()
        val w = if (shape[3] > 0) shape[3] else fallbackImgsz.toLong()

        if (c != 3L) {
            throw RuntimeException("Expected 3-channel input, got C=$c")
        }

        return OrtInputShape(n = n, c = c, h = h, w = w)
    }

    private fun loadBitmapsFromAssets(dataset: String, maxImages: Int): List<Bitmap> {
        val names = try {
            assets.list(dataset)
                ?.filter {
                    it.endsWith(".png", true) ||
                            it.endsWith(".jpg", true) ||
                            it.endsWith(".jpeg", true)
                }
                ?.sorted()
                ?.take(maxImages)
                ?: emptyList()
        } catch (_: IOException) {
            emptyList()
        }

        val out = ArrayList<Bitmap>(names.size)
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        for (fn in names) {
            try {
                assets.open("$dataset/$fn").use { stream ->
                    val bmp = BitmapFactory.decodeStream(stream, null, opts)
                    if (bmp != null) {
                        out.add(bmp)
                    }
                }
            } catch (_: Exception) {
            }
        }
        return out
    }

    private fun bitmapToFloatCHWLetterbox(src: Bitmap, targetW: Int, targetH: Int): FloatBuffer {
        val scale = min(
            targetW.toFloat() / src.width.toFloat(),
            targetH.toFloat() / src.height.toFloat()
        )

        val newW = (src.width * scale).roundToInt().coerceAtLeast(1)
        val newH = (src.height * scale).roundToInt().coerceAtLeast(1)

        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        val canvasBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val left = ((targetW - newW) / 2f)
        val top = ((targetH - newH) / 2f)
        canvas.drawBitmap(resized, left, top, null)

        val pixels = IntArray(targetW * targetH)
        canvasBitmap.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        val byteBuf = ByteBuffer
            .allocateDirect(4 * 3 * targetW * targetH)
            .order(ByteOrder.nativeOrder())

        val floatBuf = byteBuf.asFloatBuffer()
        val plane = targetW * targetH

        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255.0f
            val g = ((p shr 8) and 0xFF) / 255.0f
            val b = (p and 0xFF) / 255.0f

            floatBuf.put(i, r)
            floatBuf.put(plane + i, g)
            floatBuf.put(2 * plane + i, b)
        }

        floatBuf.rewind()
        resized.recycle()
        canvasBitmap.recycle()
        return floatBuf
    }

    private fun loadImagesFromAssets(dataset: String, maxImages: Int): List<ImageData> {
        val names = try {
            assets.list(dataset)
                ?.filter {
                    it.endsWith(".png", true) ||
                            it.endsWith(".jpg", true) ||
                            it.endsWith(".jpeg", true)
                }
                ?.sorted()
                ?.take(maxImages)
                ?: emptyList()
        } catch (_: IOException) {
            emptyList()
        }

        val out = ArrayList<ImageData>(names.size)
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        for (fn in names) {
            try {
                assets.open("$dataset/$fn").use { stream ->
                    val bmp = BitmapFactory.decodeStream(stream, null, opts) ?: return@use
                    val buf = bitmapToRgbaBuffer(bmp)
                    out.add(ImageData(buf, bmp.width, bmp.height))
                    bmp.recycle()
                }
            } catch (_: Exception) {
            }
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
            buffer.put(((pixel shr 16) and 0xFF).toByte())
            buffer.put(((pixel shr 8) and 0xFF).toByte())
            buffer.put((pixel and 0xFF).toByte())
            buffer.put(((pixel shr 24) and 0xFF).toByte())
        }

        buffer.rewind()
        return buffer
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdownNow()
    }
}