package com.example.testyolo

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity

class StartActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_start)

        findViewById<android.view.View>(R.id.btnCamera).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }

        findViewById<android.view.View>(R.id.btnBenchmark).setOnClickListener {
            startActivity(Intent(this, BenchmarkActivity::class.java))
        }

        findViewById<android.view.View>(R.id.btnYoloBenchmark).setOnClickListener {
            startActivity(Intent(this, YoloBenchmarkActivity::class.java))
        }

        findViewById<android.view.View>(R.id.btnYoloSegBenchmark).setOnClickListener {
            startActivity(Intent(this, YoloSegBenchmarkActivity::class.java))
        }
    }
}

