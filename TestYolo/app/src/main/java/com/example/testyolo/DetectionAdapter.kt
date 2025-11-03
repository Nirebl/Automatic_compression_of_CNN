package com.example.testyolo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView

class DetectionAdapter : ListAdapter<DetectionEvent, DetectionAdapter.VH>(DIFF) {
    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val title: TextView = v.findViewById(R.id.tvTitle)
        val meta: TextView = v.findViewById(R.id.tvMeta)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_detection, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val e = getItem(position)
        holder.title.text = e.label
        holder.meta.text = "${e.timeString()}  •  ${e.scoreString()}"
    }

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<DetectionEvent>() {
            override fun areItemsTheSame(oldItem: DetectionEvent, newItem: DetectionEvent) =
                oldItem.tsMs == newItem.tsMs && oldItem.label == newItem.label

            override fun areContentsTheSame(oldItem: DetectionEvent, newItem: DetectionEvent) =
                oldItem == newItem
        }
    }
}
