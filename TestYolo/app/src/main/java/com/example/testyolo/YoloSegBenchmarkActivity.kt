package com.example.testyolo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class SegBenchmarkResult(
    val resolution: Int,
    val avgTimeMs: Double,
    val fps: Double,
    val numSegmentations: Int,
    val minTimeMs: Double,
    val maxTimeMs: Double,
    val avgMaskPixels: Int
)

class YoloSegBenchmarkActivity : ComponentActivity() {

    // Available input resolutions (must have yolov11n-seg_{size}.param/bin in assets)
    // 320, 480 use their own models; 640+ all use the 640 model
    private val resolutions = listOf(320, 480, 640, 768, 896, 1024, 1280)
    private val selectedResolutions = mutableSetOf(320, 480, 640, 1024)

    // UI
    private lateinit var tvStatus: TextView
    private lateinit var tvTotalTime: TextView
    private lateinit var tvDatasetInfo: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnRun: Button
    private lateinit var btnStop: Button
    private lateinit var rvResults: RecyclerView
    private lateinit var spinnerIterations: Spinner
    private lateinit var btnSelectResolutions: Button
    private lateinit var tvSelectedResolutions: TextView

    private val resultsAdapter = SegResolutionResultAdapter()
    private val executor = Executors.newSingleThreadExecutor()
    private val isRunning = AtomicBoolean(false)
    private val shouldStop = AtomicBoolean(false)

    private var iterations = 10

    // JNI Bridge for YOLOv11-seg with configurable input size
    object YoloSegBridge {
        init {
            System.loadLibrary("ncnn")
            System.loadLibrary("yolo")
        }

        external fun initSeg(assetMgr: android.content.res.AssetManager): Boolean
        external fun loadForSize(inputSize: Int): Boolean  // Load model for specific size
        external fun getLoadedSize(): Int  // Get currently loaded model size
        external fun detectSegRgbaWithSize(
            rgba: ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int,
            conf: Float, iou: Float, inputSize: Int
        ): Array<FloatArray>
        external fun releaseSeg()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_yoloseg_benchmark)

        // Bind views
        tvStatus = findViewById(R.id.tvStatus)
        tvTotalTime = findViewById(R.id.tvTotalTime)
        tvDatasetInfo = findViewById(R.id.tvDatasetInfo)
        progressBar = findViewById(R.id.progressBar)
        btnRun = findViewById(R.id.btnRun)
        btnStop = findViewById(R.id.btnStop)
        rvResults = findViewById(R.id.rvResults)
        spinnerIterations = findViewById(R.id.spinnerIterations)
        btnSelectResolutions = findViewById(R.id.btnSelectResolutions)
        tvSelectedResolutions = findViewById(R.id.tvSelectedResolutions)

        findViewById<ImageButton>(R.id.btnBack).setOnClickListener { finish() }

        // Setup iterations spinner
        val iterationOptions = listOf(1, 5, 10, 20, 50, 100)
        spinnerIterations.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            iterationOptions.map { if (it == 1) "1 iteration" else "$it iterations" }
        )
        spinnerIterations.setSelection(2) // default 10
        spinnerIterations.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                iterations = iterationOptions[position]
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Setup resolution selector
        updateSelectedResolutionsText()
        btnSelectResolutions.setOnClickListener { showResolutionDialog() }

        // Setup RecyclerView
        rvResults.layoutManager = LinearLayoutManager(this)
        rvResults.adapter = resultsAdapter

        btnRun.setOnClickListener { startBenchmark() }
        btnStop.setOnClickListener { stopBenchmark() }

        // Initialize JNI (models will be loaded per resolution)
        executor.execute {
            val ok = YoloSegBridge.initSeg(assets)
            runOnUiThread {
                if (ok) {
                    tvStatus.text = "Ready. Select resolutions and run benchmark.\n(Requires yolov11n-seg_XXX.param/bin files)"
                } else {
                    tvStatus.text = "Failed to initialize!"
                    btnRun.isEnabled = false
                }
            }
        }
    }

    private fun showResolutionDialog() {
        val resolutionLabels = resolutions.map { "${it}px" }.toTypedArray()
        val checkedItems = resolutions.map { selectedResolutions.contains(it) }.toBooleanArray()

        android.app.AlertDialog.Builder(this, R.style.DarkDialogTheme)
            .setTitle("Select Resolutions")
            .setMultiChoiceItems(resolutionLabels, checkedItems) { _, which, isChecked ->
                if (isChecked) {
                    selectedResolutions.add(resolutions[which])
                } else {
                    selectedResolutions.remove(resolutions[which])
                }
            }
            .setPositiveButton("OK") { _, _ ->
                updateSelectedResolutionsText()
            }
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Select All") { _, _ ->
                selectedResolutions.clear()
                selectedResolutions.addAll(resolutions)
                updateSelectedResolutionsText()
            }
            .show()
    }

    private fun updateSelectedResolutionsText() {
        tvSelectedResolutions.text = if (selectedResolutions.isEmpty()) {
            "None selected"
        } else {
            selectedResolutions.sorted().joinToString(", ") { "${it}px" }
        }
    }

    override fun onDestroy() {
        shouldStop.set(true)
        executor.shutdown()
        YoloSegBridge.releaseSeg()
        super.onDestroy()
    }

    private fun startBenchmark() {
        if (isRunning.get()) return
        if (selectedResolutions.isEmpty()) {
            tvStatus.text = "Please select at least one resolution!"
            return
        }

        isRunning.set(true)
        shouldStop.set(false)
        btnRun.isEnabled = false
        btnStop.isEnabled = true
        resultsAdapter.submitList(emptyList())

        executor.execute { runBenchmark() }
    }

    private fun stopBenchmark() {
        shouldStop.set(true)
        btnStop.isEnabled = false
    }

    // Maximum images to load (to avoid OOM)
    private val maxImages = 10
    // Maximum dimension for loaded images (to prevent memory issues with segmentation)
    private val maxImageDimension = 640

    private fun runBenchmark() {
        runOnUiThread {
            tvStatus.text = "Loading ADE20K images..."
            progressBar.progress = 0
        }

        // Load images from assets/ade20k
        val bitmaps = mutableListOf<Bitmap>()
        try {
            val ade20kImages = assets.list("ade20k")?.filter {
                it.endsWith(".png") || it.endsWith(".jpg") || it.endsWith(".jpeg")
            }?.sorted()?.take(maxImages) ?: emptyList()

            for ((index, filename) in ade20kImages.withIndex()) {
                if (shouldStop.get()) break

                runOnUiThread {
                    tvStatus.text = "Loading image ${index + 1}/${ade20kImages.size}..."
                    progressBar.progress = (index * 20) / ade20kImages.size.coerceAtLeast(1)
                }

                try {
                    // First, decode bounds only to calculate sample size
                    val boundsOptions = BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }
                    assets.open("ade20k/$filename").use { stream ->
                        BitmapFactory.decodeStream(stream, null, boundsOptions)
                    }

                    // Calculate appropriate sample size to scale down large images
                    val sampleSize = calculateInSampleSize(
                        boundsOptions.outWidth,
                        boundsOptions.outHeight,
                        maxImageDimension,
                        maxImageDimension
                    )

                    // Now decode with the calculated sample size
                    val decodeOptions = BitmapFactory.Options().apply {
                        inSampleSize = sampleSize
                    }
                    val bmp = assets.open("ade20k/$filename").use { stream ->
                        BitmapFactory.decodeStream(stream, null, decodeOptions)
                    }
                    if (bmp != null) bitmaps.add(bmp)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }

        if (bitmaps.isEmpty()) {
            runOnUiThread {
                tvStatus.text = "No images found in assets/ade20k/"
                tvDatasetInfo.text = "Please add ADE20K images to assets/ade20k folder"
                isRunning.set(false)
                btnRun.isEnabled = true
                btnStop.isEnabled = false
            }
            return
        }

        // Show image info
        val avgWidth = bitmaps.map { it.width }.average().toInt()
        val avgHeight = bitmaps.map { it.height }.average().toInt()
        runOnUiThread {
            tvDatasetInfo.text = "${bitmaps.size} images • avg ${avgWidth}×${avgHeight}"
            tvStatus.text = "Preparing benchmark..."
        }
        Thread.sleep(300)

        // Pre-convert all bitmaps to buffers
        data class ImageData(val buffer: ByteBuffer, val width: Int, val height: Int)
        val imageDataList = bitmaps.map { bmp ->
            ImageData(bitmapToRgbaBuffer(bmp), bmp.width, bmp.height)
        }

        val sortedResolutions = selectedResolutions.sorted()
        val totalSteps = sortedResolutions.size
        val results = mutableListOf<SegBenchmarkResult>()
        val startTotal = SystemClock.elapsedRealtimeNanos()

        for ((idx, resolution) in sortedResolutions.withIndex()) {
            if (shouldStop.get()) break

            runOnUiThread {
                tvStatus.text = "Loading model for ${resolution}px..."
                progressBar.progress = (idx * 100) / totalSteps
            }

            // Load the model for this resolution
            val modelLoaded = YoloSegBridge.loadForSize(resolution)
            if (!modelLoaded) {
                runOnUiThread {
                    tvStatus.text = "Failed to load model for ${resolution}px! Skipping..."
                }
                Thread.sleep(500)
                continue
            }

            // Warm-up run
            val warmupData = imageDataList.first()
            warmupData.buffer.rewind()
            YoloSegBridge.detectSegRgbaWithSize(
                warmupData.buffer,
                warmupData.width, warmupData.height,
                warmupData.width * 4, 0, 0.25f, 0.45f, resolution
            )
            Thread.sleep(100)

            runOnUiThread {
                tvStatus.text = "Testing ${resolution}px... (${idx + 1}/$totalSteps)"
            }

            val times = mutableListOf<Double>()
            var totalSegmentations = 0
            var totalMaskPixels = 0L

            // Benchmark runs
            for (iter in 0 until iterations) {
                if (shouldStop.get()) break

                for (imgData in imageDataList) {
                    if (shouldStop.get()) break

                    imgData.buffer.rewind()

                    val start = SystemClock.elapsedRealtimeNanos()
                    val dets = YoloSegBridge.detectSegRgbaWithSize(
                        imgData.buffer,
                        imgData.width, imgData.height,
                        imgData.width * 4, 0, 0.25f, 0.45f, resolution
                    )
                    val end = SystemClock.elapsedRealtimeNanos()

                    times.add((end - start) / 1_000_000.0)
                    totalSegmentations += dets.size

                    // Sum up mask pixels (mask_w * mask_h from detection results)
                    for (det in dets) {
                        if (det.size >= 8) {
                            val maskW = det[6].toInt()
                            val maskH = det[7].toInt()
                            totalMaskPixels += (maskW * maskH).toLong()
                        }
                    }
                }
            }

            if (times.isNotEmpty()) {
                val avgTime = times.average()
                val minTime = times.minOrNull() ?: 0.0
                val maxTime = times.maxOrNull() ?: 0.0
                val fps = if (avgTime > 0) 1000.0 / avgTime else 0.0
                val avgSegs = totalSegmentations / times.size
                val avgMaskPx = if (totalSegmentations > 0) (totalMaskPixels / totalSegmentations).toInt() else 0

                results.add(SegBenchmarkResult(
                    resolution = resolution,
                    avgTimeMs = avgTime,
                    fps = fps,
                    numSegmentations = avgSegs,
                    minTimeMs = minTime,
                    maxTimeMs = maxTime,
                    avgMaskPixels = avgMaskPx
                ))

                runOnUiThread {
                    resultsAdapter.submitList(results.toList())
                }
            }
        }

        val endTotal = SystemClock.elapsedRealtimeNanos()
        val totalTimeMs = (endTotal - startTotal) / 1_000_000.0

        // Clean up bitmaps
        bitmaps.forEach { it.recycle() }
        bitmaps.clear()
        System.gc()

        runOnUiThread {
            tvTotalTime.text = formatTime(totalTimeMs)
            tvStatus.text = if (shouldStop.get()) "Stopped" else "Completed!"
            progressBar.progress = 100
            isRunning.set(false)
            btnRun.isEnabled = true
            btnStop.isEnabled = false
        }
    }

    private fun bitmapToRgbaBuffer(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val buffer = ByteBuffer.allocateDirect(width * height * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixel in pixels) {
            buffer.put((pixel shr 16 and 0xFF).toByte()) // R
            buffer.put((pixel shr 8 and 0xFF).toByte())  // G
            buffer.put((pixel and 0xFF).toByte())        // B
            buffer.put((pixel shr 24 and 0xFF).toByte()) // A
        }
        buffer.rewind()
        return buffer
    }

    private fun formatTime(ms: Double): String {
        return when {
            ms >= 1000 -> String.format(Locale.US, "%.2fs", ms / 1000)
            else -> String.format(Locale.US, "%.1fms", ms)
        }
    }

    private fun calculateInSampleSize(width: Int, height: Int, reqWidth: Int, reqHeight: Int): Int {
        // Find the smallest power of 2 that scales the image down to fit within reqWidth x reqHeight
        var inSampleSize = 1
        while (width / inSampleSize > reqWidth || height / inSampleSize > reqHeight) {
            inSampleSize *= 2
        }
        return inSampleSize
    }
}

// Adapter for segmentation benchmark results
class SegResolutionResultAdapter : ListAdapter<SegBenchmarkResult, SegResolutionResultAdapter.VH>(DIFF) {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val tvResolution: TextView = view.findViewById(R.id.tvResolution)
        val tvFps: TextView = view.findViewById(R.id.tvFps)
        val tvLatency: TextView = view.findViewById(R.id.tvLatency)
        val tvDetails: TextView = view.findViewById(R.id.tvDetails)
        val tvMaskInfo: TextView = view.findViewById(R.id.tvMaskInfo)
        val progressFps: ProgressBar = view.findViewById(R.id.progressFps)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_seg_result, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = getItem(position)
        holder.tvResolution.text = "${item.resolution}"
        holder.tvFps.text = String.format(Locale.US, "%.1f FPS", item.fps)
        holder.tvLatency.text = String.format(Locale.US, "%.1f ms", item.avgTimeMs)
        holder.tvDetails.text = String.format(
            Locale.US, "min: %.1fms | max: %.1fms",
            item.minTimeMs, item.maxTimeMs
        )
        holder.tvMaskInfo.text = String.format(
            Locale.US, "~%d segs • %d mask px",
            item.numSegmentations, item.avgMaskPixels
        )

        // FPS bar (normalize to 0-30 FPS range for segmentation which is slower)
        holder.progressFps.progress = (item.fps * 100 / 30).toInt().coerceIn(0, 100)
    }

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<SegBenchmarkResult>() {
            override fun areItemsTheSame(old: SegBenchmarkResult, new: SegBenchmarkResult) =
                old.resolution == new.resolution
            override fun areContentsTheSame(old: SegBenchmarkResult, new: SegBenchmarkResult) =
                old == new
        }
    }
}
