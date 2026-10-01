package com.thesis.t2520738
import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import ai.onnxruntime.providers.NNAPIFlags
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.max
import kotlin.math.min
class YoloDetector(private val ctx:Context){
 companion object { val NAMES=arrayOf("bicycle","bus","car","cng","cones","dog","leguna","motorcycle","mpv","pedestrian","pickup","rickshaw","roadDivider","trafficLight","trafficSign","trees","truck","van","wheelBarrow") }
 private val env=OrtEnvironment.getEnvironment(); private var session:OrtSession?=null; var mode="CPU"; private set
 fun setMode(npu:Boolean):String { session?.close(); val o=OrtSession.SessionOptions(); mode=if(npu) "NPU" else "CPU"
  if(npu) o.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
  val bytes=ctx.assets.open("best.onnx").readBytes(); session=env.createSession(bytes,o); o.close(); return if(npu) "NNAPI ACTIVE (CPU disabled in NNAPI)" else "CPU"
 }
 fun detect(src:Bitmap):Pair<List<Detection>,Double>{ if(session==null)setMode(false); val b=Bitmap.createScaledBitmap(src,640,640,true); val px=IntArray(640*640);b.getPixels(px,0,640,0,0,640,640); val data=FloatArray(3*640*640); val plane=640*640
  for(i in px.indices){val v=px[i];data[i]=((v shr 16) and 255)/255f;data[plane+i]=((v shr 8) and 255)/255f;data[2*plane+i]=(v and 255)/255f}
  val tensor=OnnxTensor.createTensor(env,FloatBuffer.wrap(data), longArrayOf(1,3,640,640)); val st=System.nanoTime(); val r=session!!.run(mapOf(session!!.inputNames.first() to tensor)); val ms=(System.nanoTime()-st)/1e6; val out=r[0].value as Array<Array<FloatArray>>; val a=out[0]; val cand=ArrayList<Detection>()
  for(i in 0 until 8400){var best=0f;var ci=0;for(c in 0 until 19){val s=a[4+c][i];if(s>best){best=s;ci=c}};if(best>=0.35f){val cx=a[0][i];val cy=a[1][i];val w=a[2][i];val h=a[3][i];cand.add(Detection(cx-w/2,cy-h/2,cx+w/2,cy+h/2,best,ci))}}
  tensor.close();r.close(); return Pair(nms(cand,0.45f),ms)
 }
 private fun nms(x:List<Detection>,thr:Float):List<Detection>{val out=ArrayList<Detection>();for(c in 0 until 19){val q=x.filter{it.classId==c}.sortedByDescending{it.score}.toMutableList();while(q.isNotEmpty()){val a=q.removeAt(0);out.add(a);q.removeAll{iou(a,it)>thr}}};return out}
 private fun iou(a:Detection,b:Detection):Float{val x1=max(a.x1,b.x1);val y1=max(a.y1,b.y1);val x2=min(a.x2,b.x2);val y2=min(a.y2,b.y2);val inter=max(0f,x2-x1)*max(0f,y2-y1);val aa=max(0f,a.x2-a.x1)*max(0f,a.y2-a.y1);val bb=max(0f,b.x2-b.x1)*max(0f,b.y2-b.y1);return inter/(aa+bb-inter+1e-6f)}
}
