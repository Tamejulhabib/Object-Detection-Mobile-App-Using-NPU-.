package com.thesis.t2520738
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
class OverlayView(c:Context,a:AttributeSet?):View(c,a){
 private var ds:List<Detection> = emptyList(); private var iw=640; private var ih=640
 private val p=Paint().apply{style=Paint.Style.STROKE;strokeWidth=4f;color=Color.GREEN}
 private val t=Paint().apply{style=Paint.Style.FILL;textSize=32f;color=Color.WHITE}
 fun setDetections(x:List<Detection>,w:Int,h:Int){ds=x;iw=w;ih=h;postInvalidate()}
 override fun onDraw(c:Canvas){super.onDraw(c); val sx=width.toFloat()/iw; val sy=height.toFloat()/ih
  ds.forEach{d-> c.drawRect(d.x1*sx,d.y1*sy,d.x2*sx,d.y2*sy,p); c.drawText("${YoloDetector.NAMES[d.classId]} ${(d.score*100).toInt()}%",d.x1*sx,(d.y1*sy-8).coerceAtLeast(30f),t)} }
}
