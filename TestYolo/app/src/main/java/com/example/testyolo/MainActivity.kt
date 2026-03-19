package com.example.testyolo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import android.view.Surface
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.tabs.TabLayout
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

enum class Mode { YOLO, YOLOSEG, RESNET }

class MainActivity : ComponentActivity() {

    // --- UI ---
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var hud: TextView
    private lateinit var spModel: Spinner

    private lateinit var tabLayout: TabLayout
    private lateinit var liveContainer: View
    private lateinit var logContainer: View
    private lateinit var rvLog: RecyclerView
    private lateinit var btnClearLog: View
    private val logAdapter = DetectionAdapter()
    private var logListener: ((List<DetectionEvent>) -> Unit)? = null

    // Labels
    private lateinit var labels: List<String>

    // Camera / analyzer
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val workExecutor = Executors.newSingleThreadExecutor()
    private var cameraProviderRef: ProcessCameraProvider? = null
    private var previewUseCase: Preview? = null
    private var analysisUseCase: ImageAnalysis? = null

    // Session / race protection
    @Volatile private var sessionToken = AtomicLong(0L)
    private val inFlight = AtomicInteger(0)          // кол-во активных инференсов
    private val switching = AtomicBoolean(false)     // флаг "идёт переключение"

    // FPS
    private var lastT = 0L
    private var frames = 0
    private var skipEvery = 1 // можно 2 для снижения нагрузки

    // Current mode
    private var mode: Mode = Mode.YOLO

    // --- JNI bridges ---

    object YoloSegBridge {
        init { System.loadLibrary("ncnn"); System.loadLibrary("yolo") }
        external fun init(assetMgr: android.content.res.AssetManager, param: String, bin: String): Boolean
        external fun detectRgbaBoxesOnly(
            rgba: ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int,
            conf: Float, iou: Float
        ): Array<FloatArray>  // [x1,y1,x2,y2,score,cls,mask_w,mask_h]
        external fun release()
    }

    object ResNetBridge {
        init { System.loadLibrary("ncnn"); System.loadLibrary("yolo") }
        external fun init(assetMgr: android.content.res.AssetManager, param: String, bin: String): Boolean
        external fun classifyRgba(
            rgba: ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int, topK: Int
        ): FloatArray // [cls0,prob0, cls1,prob1, ...]
        external fun release()
    }

    object YoloBridge {
        init { System.loadLibrary("ncnn"); System.loadLibrary("yolo") }
        external fun init(assetMgr: android.content.res.AssetManager): Boolean
        external fun detectRgba(
            rgba: ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int,
            conf: Float, iou: Float
        ): Array<FloatArray> // [x1,y1,x2,y2,score,cls]
        external fun release()
    }

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else finish() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // bind UI
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        hud = findViewById(R.id.hud)
        spModel = findViewById(R.id.spModel)

        tabLayout = findViewById(R.id.tabLayout)
        liveContainer = findViewById(R.id.liveContainer)
        logContainer = findViewById(R.id.logContainer)
        rvLog = findViewById(R.id.rvLog)
        btnClearLog = findViewById(R.id.btnClearLog)

        // PreviewView: TextureView-режим
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        // tabs
        tabLayout.addTab(tabLayout.newTab().setText("LIVE"))
        tabLayout.addTab(tabLayout.newTab().setText("LOG"))
        tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) { switchTab(tab.position) }
            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) { switchTab(tab.position) }
        })
        switchTab(0)

        // log
        rvLog.layoutManager = LinearLayoutManager(this)
        rvLog.adapter = logAdapter
        btnClearLog.setOnClickListener { DetectionLog.clear() }
        logListener = { list -> runOnUiThread { logAdapter.submitList(list) } }
        DetectionLog.addListener(logListener!!)

        // labels
        labels = loadLabels()
        overlay.setLabels(labels)

        // spinner
        spModel.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("YOLOv8 (detector)", "YOLOv11n-seg (segmentation)", "ResNet-50 (classifier)")
        )
        spModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                val newMode = when (position) {
                    1 -> Mode.YOLOSEG
                    2 -> Mode.RESNET
                    else -> Mode.YOLO
                }
                if (newMode != mode) {
                    mode = newMode
                    switchModeAtomic(newMode)
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>) {}
        }

        // init YOLO by default
        runCatching { YoloBridge.init(assets) }

        // camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else askCamera.launch(Manifest.permission.CAMERA)
    }

    override fun onDestroy() {
        logListener?.let { DetectionLog.removeListener(it) }
        runCatching { YoloBridge.release() }
        runCatching { YoloSegBridge.release() }
        runCatching { ResNetBridge.release() }
        cameraExecutor.shutdown()
        workExecutor.shutdown()
        super.onDestroy()
    }

    // ---------- Camera ----------

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            cameraProviderRef = cameraProvider

            val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(rotation)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            previewUseCase = preview

            val myToken = sessionToken.incrementAndGet()
            val analysis = buildAnalyzerUseCase(rotation, myToken)
            analysisUseCase = analysis

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun buildAnalyzerUseCase(targetRotation: Int, token: Long): ImageAnalysis {
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setTargetResolution(Size(1280, 720))
            .setTargetRotation(targetRotation)
            .build()

        analysis.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { image ->
            // игнорируем кадры прошлых сессий
            if (sessionToken.get() != token) { image.close(); return@Analyzer }

            inFlight.incrementAndGet()
            try {
                frames++
                if (frames % skipEvery != 0) { image.close(); return@Analyzer }

                val plane = image.planes[0]
                val buf = plane.buffer // RGBA8888
                when (mode) {
                    Mode.YOLO -> {
                        val dets = YoloBridge.detectRgba(
                            buf, image.width, image.height, plane.rowStride,
                            image.imageInfo.rotationDegrees, 0.25f, 0.45f
                        )
                        overlay.post {
                            overlay.update(
                                image.width, image.height, dets.toList(),
                                image.imageInfo.rotationDegrees
                            )
                            updateHudFps(dets.size)
                        }
                        val events = dets.sortedByDescending { it[4] }.take(5).map { d ->
                            DetectionEvent(
                                System.currentTimeMillis(),
                                clsName(d[5].toInt()),
                                d[4]
                            )
                        }
                        DetectionLog.addAll(events)
                    }
                    Mode.YOLOSEG -> {
                        val dets = YoloSegBridge.detectRgbaBoxesOnly(
                            buf, image.width, image.height, plane.rowStride,
                            image.imageInfo.rotationDegrees, 0.25f, 0.45f
                        )
                        overlay.post {
                            overlay.updateSeg(
                                image.width, image.height, dets.toList(),
                                image.imageInfo.rotationDegrees
                            )
                            updateHudFps(dets.size)
                        }
                        val events = dets.sortedByDescending { it[4] }.take(5).map { d ->
                            DetectionEvent(
                                System.currentTimeMillis(),
                                clsName(d[5].toInt()),
                                d[4]
                            )
                        }
                        DetectionLog.addAll(events)
                    }
                    Mode.RESNET -> {
                        val top = ResNetBridge.classifyRgba(
                            buf, image.width, image.height, plane.rowStride,
                            image.imageInfo.rotationDegrees, 5
                        )
                        overlay.post {
                            overlay.update(
                                image.width, image.height, emptyList(),
                                image.imageInfo.rotationDegrees
                            )
                            val label = if (top.size >= 2) {
                                val cls = top[0].toInt()
                                val prob = top[1]
                                "${clsName(cls)} ${"%.1f".format(Locale.US, prob * 100)}%"
                            } else "--"
                            hud.text = "$label  |  ${fpsString()}"
                        }
                        val now = System.currentTimeMillis()
                        val events = mutableListOf<DetectionEvent>()
                        var i = 0
                        while (i + 1 < top.size) {
                            val cls = top[i].toInt()
                            val prob = top[i + 1]
                            events += DetectionEvent(now, clsName(cls), prob)
                            i += 2
                        }
                        DetectionLog.addAll(events)
                    }
                }
            } catch (_: Throwable) {
                // не падаем из анализатора
            } finally {
                image.close()
                inFlight.decrementAndGet()
            }
        })
        return analysis
    }

    /**
     * Ждём завершения всех текущих инференсов.
     */
    private fun waitForInFlightToDrain(timeoutMs: Long = 800): Boolean {
        val start = SystemClock.uptimeMillis()
        while (inFlight.get() > 0) {
            if (SystemClock.uptimeMillis() - start >= timeoutMs) return false
            try { Thread.sleep(10) } catch (_: InterruptedException) { break }
        }
        return true
    }

    /**
     * Атомарное переключение профиля:
     * 1) останавливаем анализ и unbindAll()
     * 2) ждём, пока обнулятся активные инференсы
     * 3) release()/init() нужной модели (в фоне)
     * 4) новый sessionToken и bind нового анализатора
     */
    private fun switchModeAtomic(newMode: Mode) {
        if (!switching.compareAndSet(false, true)) return  // уже идёт переключение

        val provider = cameraProviderRef
        val preview = previewUseCase
        if (provider == null || preview == null) {
            switching.set(false)
            return
        }

        // 1) стоп анализа и полный unbind
        runOnUiThread {
            runCatching { analysisUseCase?.clearAnalyzer() }
            runCatching { provider.unbindAll() }
            analysisUseCase = null
        }

        // 2) дождаться окончания текущих JNI-вызовов
        waitForInFlightToDrain(1000)

        // 3) release/init на ворк-исполнителе
        workExecutor.execute {
            when (newMode) {
                Mode.YOLO -> {
                    runCatching { ResNetBridge.release() }
                    runCatching { YoloSegBridge.release() }
                    runCatching { YoloBridge.init(assets) }
                }
                Mode.YOLOSEG -> {
                    runCatching { YoloBridge.release() }
                    runCatching { ResNetBridge.release() }
                    runCatching { YoloSegBridge.init(assets, "yolov11n-seg.param", "yolov11n-seg.bin") }
                }
                Mode.RESNET -> {
                    runCatching { YoloBridge.release() }
                    runCatching { YoloSegBridge.release() }
                    runCatching { ResNetBridge.init(assets, "resnet50.param", "resnet50.bin") }
                }
            }

            val myToken = sessionToken.incrementAndGet()

            // 4) пересборка анализатора и bind
            runOnUiThread {
                try {
                    val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
                    preview.setSurfaceProvider(previewView.surfaceProvider)
                    val analysis = buildAnalyzerUseCase(rotation, myToken)
                    analysisUseCase = analysis
                    cameraProviderRef?.bindToLifecycle(
                        this@MainActivity,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis
                    )
                } catch (_: Throwable) {
                    // не роняем процесс
                } finally {
                    switching.set(false)
                }
            }
        }
    }

    // ---------- UI helpers ----------

    private fun switchTab(pos: Int) {
        if (pos == 0) {
            liveContainer.visibility = View.VISIBLE
            logContainer.visibility = View.GONE
        } else {
            liveContainer.visibility = View.GONE
            logContainer.visibility = View.VISIBLE
        }
    }

    private fun clsName(clsIdx: Int): String =
        labels.getOrNull(clsIdx) ?: "class_$clsIdx"

    private fun updateHudFps(numDet: Int) {
        val now = System.nanoTime()
        if (lastT == 0L) lastT = now
        val dt = (now - lastT) / 1e9
        if (dt >= 1.0) {
            val fps = frames / dt
            hud.text = String.format(Locale.US, "%.1f fps | %d det", fps, numDet)
            frames = 0; lastT = now
        }
    }

    private fun fpsString(): String {
        val now = System.nanoTime()
        if (lastT == 0L) return "-- fps"
        val dt = (now - lastT) / 1e9
        val fps = if (dt > 0) frames / dt else 0.0
        return String.format(Locale.US, "%.1f fps", fps)
    }

    private fun loadLabels(): List<String> {
        // coco.names
        runCatching {
            assets.open("coco.names").bufferedReader().useLines { lines ->
                val list = lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
                if (list.isNotEmpty()) return list
            }
        }
        // metadata.yaml
        runCatching {
            val yaml = assets.open("metadata.yaml").bufferedReader().use { it.readText() }
            Regex("""names:\s*\[(.*?)\]""", RegexOption.DOT_MATCHES_ALL)
                .find(yaml)?.groupValues?.getOrNull(1)?.let { inside ->
                    val list = inside.split(',').map { it.trim().trim('"', '\'') }
                    if (list.isNotEmpty()) return list
                }
            Regex("""names:\s*\{(.*?)\}""", RegexOption.DOT_MATCHES_ALL)
                .find(yaml)?.groupValues?.getOrNull(1)?.let { inside ->
                    val list = inside.split(',').mapNotNull { part ->
                        part.substringAfter(':', "").trim().trim('"', '\'').ifEmpty { null }
                    }
                    if (list.isNotEmpty()) return list
                }
        }
        return emptyList()
    }
}
