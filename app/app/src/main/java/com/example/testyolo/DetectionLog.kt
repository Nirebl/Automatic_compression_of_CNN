package com.example.testyolo

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.LinkedBlockingDeque

data class DetectionEvent(
    val tsMs: Long,
    val label: String,
    val score: Float
) {
    fun timeString(): String =
        SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault()).format(Date(tsMs))
    fun scoreString(): String =
        String.format(Locale.getDefault(), "%.1f%%", score * 100f)
}

object DetectionLog {
    private const val MAX = 500
    private val deque = LinkedBlockingDeque<DetectionEvent>(MAX)
    private val listeners = CopyOnWriteArraySet<(List<DetectionEvent>) -> Unit>()

    fun addAll(events: List<DetectionEvent>) {
        if (events.isEmpty()) return
        synchronized(this) {
            for (e in events) {
                if (deque.size == MAX) deque.pollFirst()
                deque.offerLast(e)
            }
        }
        notifyListeners()
    }

    fun clear() {
        synchronized(this) { deque.clear() }
        notifyListeners()
    }

    fun snapshot(): List<DetectionEvent> =
        synchronized(this) { deque.toList().asReversed() }

    fun addListener(l: (List<DetectionEvent>) -> Unit) { listeners.add(l); l(snapshot()) }
    fun removeListener(l: (List<DetectionEvent>) -> Unit) { listeners.remove(l) }

    private fun notifyListeners() {
        val snap = snapshot()
        for (l in listeners) l(snap)
    }
}
