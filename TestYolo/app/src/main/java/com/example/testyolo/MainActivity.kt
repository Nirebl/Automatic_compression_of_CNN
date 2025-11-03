package com.example.testyolo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.view.Surface
import android.view.View
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.tabs.TabLayout
import java.nio.ByteBuffer
import java.util.concurrent.Executors

enum class Mode { YOLO, FRCNN, RESNET }

private var mode = Mode.YOLO

class MainActivity : ComponentActivity() {

    // --- UI ---
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var hud: TextView

    // tabs & log
    private lateinit var tabLayout: TabLayout
    private lateinit var liveContainer: View
    private lateinit var logContainer: View
    private lateinit var rvLog: RecyclerView
    private lateinit var btnClearLog: View
    private val logAdapter = DetectionAdapter()
    private var logListener: ((List<DetectionEvent>) -> Unit)? = null

    // labels
    private lateinit var labels: List<String>

    // Camera / analyzer
    private val exec = Executors.newSingleThreadExecutor()
    private var lastT = 0L
    private var frames = 0
    private var skipEvery = 1 // при необходимости поставь 2 для снижения нагрузки

    object FrcnnBridge {
        init {
            System.loadLibrary("ncnn"); System.loadLibrary("yolo")
        }

        external fun init(
            assetMgr: android.content.res.AssetManager,
            param: String,
            bin: String
        ): Boolean

        external fun detectRgba(
            rgba: java.nio.ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int, conf: Float
        ): Array<FloatArray>
    }

    object ResNetBridge {
        init {
            System.loadLibrary("ncnn"); System.loadLibrary("yolo")
        }

        external fun init(
            assetMgr: android.content.res.AssetManager,
            param: String,
            bin: String
        ): Boolean

        external fun classifyRgba(
            rgba: java.nio.ByteBuffer,
            width: Int, height: Int, rowStride: Int, rotationDeg: Int, topK: Int
        ): FloatArray // [cls0,prob0, cls1,prob1, ...]
    }


    // --- JNI bridge (вложенный объект, см. имена JNI в C++) ---
    object YoloBridge {
        init {
            System.loadLibrary("ncnn") // сначала зависимость
            System.loadLibrary("yolo") // затем наша so
        }

        external fun init(assetMgr: android.content.res.AssetManager): Boolean
        external fun detectRgba(
            rgba: ByteBuffer,
            width: Int, height: Int, rowStride: Int,
            rotationDeg: Int,
            conf: Float, iou: Float
        ): Array<FloatArray> // [x1,y1,x2,y2,score,cls]
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
        tabLayout = findViewById(R.id.tabLayout)
        liveContainer = findViewById(R.id.liveContainer)
        logContainer = findViewById(R.id.logContainer)
        rvLog = findViewById(R.id.rvLog)
        btnClearLog = findViewById(R.id.btnClearLog)

        // labels & overlay
        labels = loadLabels()
        overlay.setLabels(labels)

        // init YOLO
        if (!YoloBridge.init(assets)) {
            hud.text = "YOLO init failed"
        }

        // log view setup
        rvLog.layoutManager = LinearLayoutManager(this)
        rvLog.adapter = logAdapter
        btnClearLog.setOnClickListener { DetectionLog.clear() }
        if (tabLayout.tabCount == 0) {
            tabLayout.addTab(tabLayout.newTab().setText("LIVE"))
            tabLayout.addTab(tabLayout.newTab().setText("LOG"))
        }
        tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) = switchTab(tab.position)
            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) {}
        })
        val spModel: android.widget.Spinner = findViewById(R.id.spModel)
        spModel.adapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("YOLOv8 (detector)", "Faster R-CNN (detector)", "ResNet-50 (classifier)")
        )
        spModel.onItemSelectedListener =
            object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: android.widget.AdapterView<*>,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    mode = when (position) {
                        1 -> Mode.FRCNN
                        2 -> Mode.RESNET
                        else -> Mode.YOLO
                    }
                    // Подгружаем модели по требованию
                    when (mode) {
                        Mode.YOLO -> { /* уже init сделан ниже */
                        }

                        Mode.FRCNN -> FrcnnBridge.init(assets, "fasterrcnn.param", "fasterrcnn.bin")
                        Mode.RESNET -> ResNetBridge.init(assets, "resnet50.param", "resnet50.bin")
                    }
                }

                override fun onNothingSelected(parent: android.widget.AdapterView<*>) {}
            }
        switchTab(0)
        logListener = { list -> runOnUiThread { logAdapter.submitList(list) } }
        DetectionLog.addListener(logListener!!)

        // permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
    }

    override fun onDestroy() {
        logListener?.let { DetectionLog.removeListener(it) }
        super.onDestroy()
    }

    private fun updateHudFps(numDet: Int) {
        val now = System.nanoTime()
        if (lastT == 0L) lastT = now
        val dt = (now - lastT) / 1e9
        if (dt >= 1.0) {
            val fps = frames / dt
            hud.text = String.format("%.1f fps | %d det", fps, numDet)
            frames = 0; lastT = now
        }
    }

    private fun fpsString(): String {
        val now = System.nanoTime()
        if (lastT == 0L) lastT = now
        val dt = (now - lastT) / 1e9
        return if (dt >= 1.0) {
            val fps = frames / dt; frames = 0; lastT = now
            String.format("%.1f fps", fps)
        } else "-- fps"
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val rotation = previewView.display?.rotation ?: Surface.ROTATION_0

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(rotation)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(Size(1280, 720))
                .setTargetRotation(rotation)
                .build()

            analysis.setAnalyzer(exec) { image ->
                try {
                    frames++
                    if (frames % skipEvery != 0) {
                        image.close(); return@setAnalyzer
                    }

                    val plane = image.planes[0]
                    val buf = plane.buffer

                    when (mode) {
                        Mode.YOLO -> {
                            val dets = YoloBridge.detectRgba(
                                buf, image.width, image.height, plane.rowStride,
                                image.imageInfo.rotationDegrees, 0.25f, 0.45f
                            )
                            overlay.post {
                                overlay.update(
                                    image.width,
                                    image.height,
                                    dets.toList(),
                                    image.imageInfo.rotationDegrees
                                )
                                updateHudFps(dets.size)
                            }
                            // лог (топ-5)
                            val events = dets.sortedByDescending { it[4] }.take(5).map { d ->
                                val cls = d[5].toInt(); DetectionEvent(
                                System.currentTimeMillis(),
                                clsName(cls),
                                d[4]
                            )
                            }
                            DetectionLog.addAll(events)
                        }

                        Mode.FRCNN -> {
                            val dets = FrcnnBridge.detectRgba(
                                buf, image.width, image.height, plane.rowStride,
                                image.imageInfo.rotationDegrees, 0.25f
                            )
                            overlay.post {
                                overlay.update(
                                    image.width,
                                    image.height,
                                    dets.toList(),
                                    image.imageInfo.rotationDegrees
                                )
                                updateHudFps(dets.size)
                            }
                            val events = dets.sortedByDescending { it[4] }.take(5).map { d ->
                                val cls = d[5].toInt(); DetectionEvent(
                                System.currentTimeMillis(),
                                clsName(cls),
                                d[4]
                            )
                            }
                            DetectionLog.addAll(events)
                        }

                        Mode.RESNET -> {
                            val top = ResNetBridge.classifyRgba(
                                buf, image.width, image.height, plane.rowStride,
                                image.imageInfo.rotationDegrees, 5
                            ) // top-5
                            // скрываем боксы
                            overlay.post {
                                overlay.update(
                                    image.width,
                                    image.height,
                                    emptyList(),
                                    image.imageInfo.rotationDegrees
                                )
                                // рисуем в HUD top-1
                                val label = if (top.size >= 2) {
                                    val cls = top[0].toInt();
                                    val prob = top[1]
                                    "${clsName(cls)} ${
                                        "%.1f".format(
                                            java.util.Locale.US,
                                            prob * 100
                                        )
                                    }%"
                                } else "--"
                                hud.text = "$label  |  ${fpsString()}"
                            }
                            // лог: top-5 в события
                            val now = System.currentTimeMillis()
                            val events = mutableListOf<DetectionEvent>()
                            for (i in top.indices step 2) {
                                val cls = top[i].toInt()
                                val prob = if (i + 1 < top.size) top[i + 1] else 0f
                                events += DetectionEvent(now, clsName(cls), prob)
                            }
                            DetectionLog.addAll(events)
                        }
                    }
                } finally {
                    image.close()
                }
            }


            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

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

    // Загружаем имена классов из assets:
    // 1) coco.names (по строке на класс)
    // 2) metadata.yaml (Ultralytics: names: [...] или names: {0: person, ...})
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
            // names: ["person","bicycle",...]
            Regex("""names:\s*\[(.*?)\]""", RegexOption.DOT_MATCHES_ALL)
                .find(yaml)?.groupValues?.getOrNull(1)?.let { inside ->
                    val list = inside.split(',').map { it.trim().trim('"', '\'') }
                    if (list.isNotEmpty()) return list
                }
            // names: {0: person, 1: bicycle, ...}
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
