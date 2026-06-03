package com.example.testyolo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
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

data class BenchmarkResult(
    val fileName: String,
    val prediction: String,
    val confidence: Float,
    val timeMs: Double,
    val bitmap: Bitmap?
)

class BenchmarkActivity : ComponentActivity() {

    // CIFAR-10 class names
    private val cifarClasses = listOf(
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )

    // UI
    private lateinit var tvTotalTime: TextView
    private lateinit var tvPerImage: TextView
    private lateinit var tvThroughput: TextView
    private lateinit var tvProgress: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnRun: Button
    private lateinit var btnStop: Button
    private lateinit var rvResults: RecyclerView

    private val resultsAdapter = BenchmarkResultAdapter()
    private val executor = Executors.newSingleThreadExecutor()
    private val isRunning = AtomicBoolean(false)
    private val shouldStop = AtomicBoolean(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_benchmark)

        // Bind views
        tvTotalTime = findViewById(R.id.tvTotalTime)
        tvPerImage = findViewById(R.id.tvPerImage)
        tvThroughput = findViewById(R.id.tvThroughput)
        tvProgress = findViewById(R.id.tvProgress)
        progressBar = findViewById(R.id.progressBar)
        btnRun = findViewById(R.id.btnRun)
        btnStop = findViewById(R.id.btnStop)
        rvResults = findViewById(R.id.rvResults)

        findViewById<ImageButton>(R.id.btnBack).setOnClickListener { finish() }

        rvResults.layoutManager = LinearLayoutManager(this)
        rvResults.adapter = resultsAdapter

        btnRun.setOnClickListener { startBenchmark() }
        btnStop.setOnClickListener { stopBenchmark() }

        // Initialize model (reuse JNI bridge from MainActivity)
        executor.execute {
            val ok = MainActivity.ResNetBridge.init(assets, "resnet50.param", "resnet50.bin")
            runOnUiThread {
                if (ok) {
                    tvProgress.text = "Model loaded. Ready to run."
                } else {
                    tvProgress.text = "Failed to load model!"
                    btnRun.isEnabled = false
                }
            }
        }
    }

    override fun onDestroy() {
        shouldStop.set(true)
        executor.shutdown()
        super.onDestroy()
    }

    private fun startBenchmark() {
        if (isRunning.get()) return

        isRunning.set(true)
        shouldStop.set(false)
        btnRun.isEnabled = false
        btnStop.isEnabled = true
        resultsAdapter.submitList(emptyList())

        executor.execute {
            runBenchmark()
        }
    }

    private fun stopBenchmark() {
        shouldStop.set(true)
        btnStop.isEnabled = false
    }

    private fun runBenchmark() {
        // Load CIFAR images from assets/cifar folder
        val imageFiles = try {
            assets.list("cifar")?.filter {
                it.endsWith(".png") || it.endsWith(".jpg") || it.endsWith(".bmp")
            } ?: emptyList()
        } catch (e: IOException) {
            emptyList()
        }

        if (imageFiles.isEmpty()) {
            runOnUiThread {
                tvProgress.text = "No images found in assets/cifar/"
                isRunning.set(false)
                btnRun.isEnabled = true
                btnStop.isEnabled = false
            }
            return
        }

        val results = mutableListOf<BenchmarkResult>()
        val totalImages = imageFiles.size
        var processedCount = 0
        val startTimeTotal = SystemClock.elapsedRealtimeNanos()

        for (fileName in imageFiles) {
            if (shouldStop.get()) break

            try {
                // Load image
                val inputStream = assets.open("cifar/$fileName")
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream.close()

                if (bitmap == null) continue

                // Convert to RGBA buffer
                val width = bitmap.width
                val height = bitmap.height
                val rgbaBuffer = ByteBuffer.allocateDirect(width * height * 4)
                rgbaBuffer.order(ByteOrder.nativeOrder())
                
                val pixels = IntArray(width * height)
                bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
                
                for (pixel in pixels) {
                    rgbaBuffer.put((pixel shr 16 and 0xFF).toByte()) // R
                    rgbaBuffer.put((pixel shr 8 and 0xFF).toByte())  // G
                    rgbaBuffer.put((pixel and 0xFF).toByte())        // B
                    rgbaBuffer.put((pixel shr 24 and 0xFF).toByte()) // A
                }
                rgbaBuffer.rewind()

                // Run inference with timing
                val startTime = SystemClock.elapsedRealtimeNanos()
                val result = MainActivity.ResNetBridge.classifyRgba(
                    rgbaBuffer, width, height, width * 4, 0, 5
                )
                val endTime = SystemClock.elapsedRealtimeNanos()
                val timeMs = (endTime - startTime) / 1_000_000.0

                // Parse result
                val prediction = if (result.size >= 2) {
                    val classIdx = result[0].toInt()
                    val confidence = result[1]
                    val className = if (classIdx < cifarClasses.size) {
                        cifarClasses[classIdx]
                    } else {
                        "class_$classIdx"
                    }
                    BenchmarkResult(fileName, className, confidence, timeMs, bitmap)
                } else {
                    BenchmarkResult(fileName, "unknown", 0f, timeMs, bitmap)
                }

                results.add(prediction)
                processedCount++

                // Update UI
                val progress = (processedCount * 100) / totalImages
                val currentResults = results.toList()
                runOnUiThread {
                    progressBar.progress = progress
                    tvProgress.text = "Processing: $processedCount / $totalImages"
                    resultsAdapter.submitList(currentResults)
                    rvResults.scrollToPosition(currentResults.size - 1)
                }

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        val endTimeTotal = SystemClock.elapsedRealtimeNanos()
        val totalTimeMs = (endTimeTotal - startTimeTotal) / 1_000_000.0
        val avgTimeMs = if (processedCount > 0) totalTimeMs / processedCount else 0.0
        val throughput = if (totalTimeMs > 0) (processedCount * 1000.0) / totalTimeMs else 0.0

        runOnUiThread {
            tvTotalTime.text = formatTime(totalTimeMs)
            tvPerImage.text = formatTime(avgTimeMs)
            tvThroughput.text = String.format(Locale.US, "%.1f/s", throughput)
            tvProgress.text = if (shouldStop.get()) "Stopped" else "Completed: $processedCount images"
            progressBar.progress = 100
            isRunning.set(false)
            btnRun.isEnabled = true
            btnStop.isEnabled = false
        }
    }

    private fun formatTime(ms: Double): String {
        return when {
            ms >= 1000 -> String.format(Locale.US, "%.2fs", ms / 1000)
            ms >= 1 -> String.format(Locale.US, "%.1fms", ms)
            else -> String.format(Locale.US, "%.2fms", ms)
        }
    }
}

// Adapter for benchmark results
class BenchmarkResultAdapter : ListAdapter<BenchmarkResult, BenchmarkResultAdapter.VH>(DIFF) {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val ivPreview: ImageView = view.findViewById(R.id.ivPreview)
        val tvFileName: TextView = view.findViewById(R.id.tvFileName)
        val tvPrediction: TextView = view.findViewById(R.id.tvPrediction)
        val tvTime: TextView = view.findViewById(R.id.tvTime)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_benchmark_result, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = getItem(position)
        holder.tvFileName.text = item.fileName
        holder.tvPrediction.text = "${item.prediction} (${String.format(Locale.US, "%.1f%%", item.confidence * 100)})"
        holder.tvTime.text = String.format(Locale.US, "%.1fms", item.timeMs)
        
        if (item.bitmap != null) {
            holder.ivPreview.setImageBitmap(item.bitmap)
        } else {
            holder.ivPreview.setImageResource(android.R.drawable.ic_menu_gallery)
        }
    }

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<BenchmarkResult>() {
            override fun areItemsTheSame(old: BenchmarkResult, new: BenchmarkResult) =
                old.fileName == new.fileName
            override fun areContentsTheSame(old: BenchmarkResult, new: BenchmarkResult) =
                old == new
        }
    }
}

