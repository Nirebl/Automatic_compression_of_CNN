package com.example.testyolo

import android.app.AlertDialog
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import android.widget.Switch
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

data class ResolutionBenchmarkResult(
    val resolution: Int,
    val avgTimeMs: Double,
    val fps: Double,
    val numDetections: Int,
    val minTimeMs: Double,
    val maxTimeMs: Double
)

class YoloBenchmarkActivity : ComponentActivity() {

    // Available input resolutions
    // 320, 480 use their own models; 640+ all use the 640 model
    private val resolutions = listOf(320, 480, 640, 768, 896, 1024, 1280)
    private val selectedResolutions = mutableSetOf(320, 480, 640, 1024)

    // UI
    private lateinit var tvStatus: TextView
    private lateinit var tvTotalTime: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnRun: Button
    private lateinit var btnStop: Button
    private lateinit var rvResults: RecyclerView
    private lateinit var btnSelectResolutions: Button
    private lateinit var tvSelectedResolutions: TextView
    private lateinit var spinnerIterations: Spinner
    private lateinit var switchOptimization: Switch
    private lateinit var tvOptimizationInfo: TextView

    private val resultsAdapter = ResolutionResultAdapter()
    private val executor = Executors.newSingleThreadExecutor()
    private val isRunning = AtomicBoolean(false)
    private val shouldStop = AtomicBoolean(false)

    private var iterations = 10

    // JNI Bridge
    object YoloBridge {
        init { System.loadLibrary("ncnn"); System.loadLibrary("yolo") }
        external fun init(assetMgr: android.content.res.AssetManager): Boolean
        external fun loadForSize(inputSize: Int): Boolean  // Load model for specific size
        external fun getLoadedSize(): Int  // Get currently loaded model size
        external fun detectRgbaWithSize(
            rgba: ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int,
            conf: Float, iou: Float, inputSize: Int
        ): Array<FloatArray>
        external fun release()
        
        // Optimization mode control
        external fun setOptimized(enabled: Boolean)
        external fun isOptimized(): Boolean
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_yolo_benchmark)

        // Bind views
        tvStatus = findViewById(R.id.tvStatus)
        tvTotalTime = findViewById(R.id.tvTotalTime)
        progressBar = findViewById(R.id.progressBar)
        btnRun = findViewById(R.id.btnRun)
        btnStop = findViewById(R.id.btnStop)
        rvResults = findViewById(R.id.rvResults)
        btnSelectResolutions = findViewById(R.id.btnSelectResolutions)
        tvSelectedResolutions = findViewById(R.id.tvSelectedResolutions)
        spinnerIterations = findViewById(R.id.spinnerIterations)
        switchOptimization = findViewById(R.id.switchOptimization)
        tvOptimizationInfo = findViewById(R.id.tvOptimizationInfo)

        findViewById<ImageButton>(R.id.btnBack).setOnClickListener { finish() }
        
        // Setup optimization toggle
        switchOptimization.setOnCheckedChangeListener { _, isChecked ->
            YoloBridge.setOptimized(isChecked)
            tvOptimizationInfo.text = if (isChecked) {
                "FP16, Winograd, SGEMM, Packing"
            } else {
                "Baseline mode (no optimizations)"
            }
        }

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
            val ok = YoloBridge.init(assets)
            runOnUiThread {
                if (ok) {
                    tvStatus.text = "Ready. Select resolutions and run benchmark.\n(Requires yolov8n_XXX.param/bin files)"
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

        AlertDialog.Builder(this, R.style.DarkDialogTheme)
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
        YoloBridge.release()
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

    private fun runBenchmark() {
        runOnUiThread {
            tvStatus.text = "Loading COCO images..."
            progressBar.progress = 0
        }

        // Load limited number of images from assets/coco (to avoid OOM)
        val bitmaps = mutableListOf<Bitmap>()
        try {
            val cocoImages = assets.list("coco")?.filter { 
                it.endsWith(".png") || it.endsWith(".jpg") || it.endsWith(".jpeg")
            }?.sorted()?.take(maxImages) ?: emptyList()
            
            // Use options to load scaled-down images if they're too large
            val options = BitmapFactory.Options().apply {
                inSampleSize = 1  // No scaling by default
            }
            
            for ((index, filename) in cocoImages.withIndex()) {
                if (shouldStop.get()) break
                
                runOnUiThread {
                    tvStatus.text = "Loading image ${index + 1}/${cocoImages.size}..."
                    progressBar.progress = (index * 30) / cocoImages.size.coerceAtLeast(1)
                }
                
                try {
                    val stream = assets.open("coco/$filename")
                    val bmp = BitmapFactory.decodeStream(stream, null, options)
                    stream.close()
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
                tvStatus.text = "No images found in assets/coco/"
                isRunning.set(false)
                btnRun.isEnabled = true
                btnStop.isEnabled = false
            }
            return
        }

        // Show image info and mode
        val avgWidth = bitmaps.map { it.width }.average().toInt()
        val avgHeight = bitmaps.map { it.height }.average().toInt()
        val modeStr = if (YoloBridge.isOptimized()) "OPTIMIZED" else "BASELINE"
        runOnUiThread {
            tvStatus.text = "[$modeStr] Loaded ${bitmaps.size} images (${avgWidth}x${avgHeight})"
        }
        Thread.sleep(500)

        // Pre-convert all bitmaps to buffers (to avoid allocation during timing)
        data class ImageData(val buffer: ByteBuffer, val width: Int, val height: Int)
        val imageDataList = bitmaps.map { bmp ->
            ImageData(bitmapToRgbaBuffer(bmp), bmp.width, bmp.height)
        }

        val sortedResolutions = selectedResolutions.sorted()
        val totalSteps = sortedResolutions.size
        val results = mutableListOf<ResolutionBenchmarkResult>()
        val startTotal = SystemClock.elapsedRealtimeNanos()

        for ((idx, resolution) in sortedResolutions.withIndex()) {
            if (shouldStop.get()) break

            runOnUiThread {
                tvStatus.text = "Loading model for ${resolution}px..."
                progressBar.progress = (idx * 100) / totalSteps
            }

            // Load the model for this resolution (yolov8n_320.param/bin, etc.)
            val modelLoaded = YoloBridge.loadForSize(resolution)
            if (!modelLoaded) {
                runOnUiThread {
                    tvStatus.text = "Failed to load model for ${resolution}px! Skipping..."
                }
                Thread.sleep(500)
                continue
            }

            // Warm-up run after loading new model
            val warmupData = imageDataList.first()
            warmupData.buffer.rewind()
            YoloBridge.detectRgbaWithSize(
                warmupData.buffer,
                warmupData.width, warmupData.height,
                warmupData.width * 4, 0, 0.25f, 0.45f, resolution
            )
            Thread.sleep(100) // Brief pause after warm-up

            runOnUiThread {
                tvStatus.text = "Testing ${resolution}px... (${idx + 1}/$totalSteps)"
            }

            val times = mutableListOf<Double>()
            var totalDetections = 0

            // Benchmark runs
            for (iter in 0 until iterations) {
                if (shouldStop.get()) break

                for (imgData in imageDataList) {
                    if (shouldStop.get()) break

                    imgData.buffer.rewind()  // Reset buffer position
                    
                    val start = SystemClock.elapsedRealtimeNanos()
                    val dets = YoloBridge.detectRgbaWithSize(
                        imgData.buffer,
                        imgData.width, imgData.height,
                        imgData.width * 4, 0, 0.25f, 0.45f, resolution
                    )
                    val end = SystemClock.elapsedRealtimeNanos()
                    
                    times.add((end - start) / 1_000_000.0)
                    totalDetections += dets.size
                }
            }

            if (times.isNotEmpty()) {
                val avgTime = times.average()
                val minTime = times.minOrNull() ?: 0.0
                val maxTime = times.maxOrNull() ?: 0.0
                val fps = if (avgTime > 0) 1000.0 / avgTime else 0.0

                results.add(ResolutionBenchmarkResult(
                    resolution = resolution,
                    avgTimeMs = avgTime,
                    fps = fps,
                    numDetections = totalDetections / times.size,
                    minTimeMs = minTime,
                    maxTimeMs = maxTime
                ))

                runOnUiThread {
                    resultsAdapter.submitList(results.toList())
                }
            }
        }

        val endTotal = SystemClock.elapsedRealtimeNanos()
        val totalTimeMs = (endTotal - startTotal) / 1_000_000.0

        // Clean up bitmaps to free memory
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
}

// Adapter for results
class ResolutionResultAdapter : ListAdapter<ResolutionBenchmarkResult, ResolutionResultAdapter.VH>(DIFF) {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val tvResolution: TextView = view.findViewById(R.id.tvResolution)
        val tvFps: TextView = view.findViewById(R.id.tvFps)
        val tvLatency: TextView = view.findViewById(R.id.tvLatency)
        val tvDetails: TextView = view.findViewById(R.id.tvDetails)
        val progressFps: ProgressBar = view.findViewById(R.id.progressFps)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_resolution_result, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = getItem(position)
        holder.tvResolution.text = "${item.resolution}"
        holder.tvFps.text = String.format(Locale.US, "%.1f FPS", item.fps)
        holder.tvLatency.text = String.format(Locale.US, "%.1f ms", item.avgTimeMs)
        holder.tvDetails.text = String.format(
            Locale.US, "min: %.1fms | max: %.1fms | ~%d det",
            item.minTimeMs, item.maxTimeMs, item.numDetections
        )
        
        // FPS bar (normalize to 0-60 FPS range)
        holder.progressFps.progress = (item.fps * 100 / 60).toInt().coerceIn(0, 100)
    }

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<ResolutionBenchmarkResult>() {
            override fun areItemsTheSame(old: ResolutionBenchmarkResult, new: ResolutionBenchmarkResult) =
                old.resolution == new.resolution
            override fun areContentsTheSame(old: ResolutionBenchmarkResult, new: ResolutionBenchmarkResult) =
                old == new
        }
    }
}


