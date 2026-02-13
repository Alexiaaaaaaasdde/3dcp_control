#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema 3DCP Integrado - Versión Jetson Nano Producción
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from datetime import datetime
from collections import deque

# Fix para Jetson
os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['CUDA_CACHE_DISABLE'] = '0'

# Detectar GPU
VPI_AVAILABLE = False
CUDA_AVAILABLE = False
try:
    import vpi
    VPI_AVAILABLE = True
    print("[GPU] ✓ VPI disponible")
except ImportError:
    pass

try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_AVAILABLE = True
        print(f"[GPU] ✓ CUDA disponible")
except:
    pass

# Imports del proyecto
from config import HW, CTRL, VISION, LOG
from utils.safety import SafetySystem, EstadoSistema
from controllers.health_monitor import HealthMonitor, ComponenteSalud
from controllers.pid_concrete import ConcretePIDController
from sensors.dual_tof import DualToFSensor, TOF_AVAILABLE

# Intentar importar Orbbec
try:
    from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat, VideoStreamProfile
    ORBBEC_AVAILABLE = True
except ImportError:
    ORBBEC_AVAILABLE = False
    print("[ERROR] pyorbbecsdk no disponible")

# ==================== PROCESADORES GPU (de tu código original) ====================

class ProcesadorCPU:
    def __init__(self, width=320, height=288):
        self.width, self.height = width, height
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        print(f"[CPU] Procesador inicializado: {width}x{height}")
        
    def procesar(self, gray_image):
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)

class ProcesadorGPU_VPI:
    def __init__(self, width=320, height=288):
        self.width, self.height = width, height
        print(f"[VPI] Procesador inicializado: {width}x{height}")
        
    def procesar(self, gray_image):
        try:
            if gray_image.dtype != np.uint8:
                gray_image = gray_image.astype(np.uint8)
            gray_image = np.ascontiguousarray(gray_image)
            with vpi.Backend.CUDA:
                vpi_img = vpi.asimage(gray_image)
                vpi_blur = vpi_img.gaussian_filter(5, border=vpi.Border.ZERO)
                result = vpi_blur.cpu()
            return result
        except Exception as e:
            print(f"[VPI] Error: {e}")
            return gray_image

# ==================== DETECTOR DE FILAMENTO (adaptado de tu código) ====================

def calcular_ancho_robusto(contorno, metodo="ELLIPSE"):
    try:
        if metodo == "ELLIPSE" and len(contorno) >= 5:
            ellipse = cv2.fitEllipse(contorno)
            eje_mayor, eje_menor = ellipse[1]
            return min(eje_mayor, eje_menor)
        elif metodo == "DISTANCE":
            x, y, w, h = cv2.boundingRect(contorno)
            mask = np.zeros((h + 10, w + 10), dtype=np.uint8)
            cv2.drawContours(mask, [contorno - [x - 5, y - 5]], -1, 255, -1)
            return 2.0 * np.max(cv2.distanceTransform(mask, cv2.DIST_L2, 5))
    except Exception:
        return None

class MedidorAnchoFilamento:
    """Versión simplificada pero robusta de tu MedidorAnchoGPU"""
    
    def __init__(self, px_to_mm=0.075):
        self.px_to_mm = px_to_mm
        self.buffer_anchos = deque(maxlen=VISION.FILTER_WINDOW)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (VISION.MORPH_KERNEL_SIZE, VISION.MORPH_KERNEL_SIZE))
        
        # Seleccionar procesador
        if VPI_AVAILABLE:
            self.procesador = ProcesadorGPU_VPI(640, 576)
            self.modo = "VPI-GPU"
        else:
            self.procesador = ProcesadorCPU(640, 576)
            self.modo = "CPU"
            
        self.consecutive_failures = 0
        self.FAILURE_THRESHOLD = VISION.FAILURE_THRESHOLD
        
        print(f"[MEDIDOR] Modo: {self.modo}")
        
    def procesar(self, color_frame, z_mm=None):
        try:
            # Extraer imagen BGR del frame Orbbec
            if hasattr(color_frame, 'get_data'):
                color_data = np.copy(np.asanyarray(color_frame.get_data()))
                fmt = str(color_frame.get_format()).upper()
                
                if "MJPG" in fmt or "MJPEG" in fmt:
                    bgr = cv2.imdecode(color_data.flatten(), cv2.IMREAD_COLOR)
                elif color_data.ndim == 3 and color_data.shape[2] == 3:
                    bgr = color_data
                elif color_data.ndim == 3 and color_data.shape[2] == 4:
                    bgr = cv2.cvtColor(color_data, cv2.COLOR_BGRA2BGR)
                else:
                    bgr = cv2.imdecode(color_data.flatten(), cv2.IMREAD_COLOR)
            else:
                return None, None, None, True
                
            if bgr is None:
                self.consecutive_failures += 1
                return None, None, None, self.consecutive_failures >= self.FAILURE_THRESHOLD
                
            h, w = bgr.shape[:2]
            
            # ROI
            if VISION.ROI_ENABLE:
                rw = int(w * VISION.ROI_WIDTH_PERCENT)
                rh = int(h * VISION.ROI_HEIGHT_PERCENT)
                x1, y1 = (w - rw) // 2, (h - rh) // 2
                x2, y2 = x1 + rw, y1 + rh
                roi = bgr[y1:y2, x1:x2].copy()
            else:
                x1, y1, x2, y2 = 0, 0, w, h
                roi = bgr.copy()
                
            # Procesamiento
            roi_h, roi_w = roi.shape[:2]
            scale = 1
            if VISION.PROCESS_HALF_RES and roi_w > 40:
                roi = cv2.resize(roi, (roi_w // 2, roi_h // 2), interpolation=cv2.INTER_LINEAR)
                scale = 2
                
            # Pipeline de visión
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = self.procesador.procesar(gray)
            
            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
            # CLAHE para mejor contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.bilateralFilter(gray, 5, 50, 50)
            
            # Binarización Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morfología
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            
            # Restaurar escala para máscara
            mask_binaria = binary.copy()
            if scale == 2:
                mask_binaria = cv2.resize(mask_binaria, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar por forma de filamento
            min_area = max(10, VISION.MIN_AREA_CONTORNO // (scale * scale))
            candidatos = []
            
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                    
                rect = cv2.minAreaRect(c)
                rw_c, rh_c = rect[1]
                if rw_c <= 0 or rh_c <= 0:
                    continue
                    
                aspect = max(rw_c, rh_c) / min(rw_c, rh_c)
                if not (VISION.MIN_ASPECT_RATIO <= aspect <= VISION.MAX_ASPECT_RATIO):
                    continue
                    
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and area / hull_area < VISION.MIN_SOLIDITY:
                    continue
                    
                candidatos.append((c, rect, area))
                
            # Visualización
            vis = bgr.copy()
            if VISION.ROI_ENABLE:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
            ancho_mm = None
            ancho_px = None
            
            if candidatos:
                # Seleccionar el más grande
                cont, rect, area = max(candidatos, key=lambda x: x[2])
                
                # Calcular ancho robusto
                ancho_px = calcular_ancho_robusto(cont, "ELLIPSE") or min(rect[1])
                ancho_px *= scale
                
                # Convertir a mm
                if z_mm and z_mm > 0:
                    # Compensación por altura
                    factor = self.px_to_mm * (HW.ALTURA_REF_MM / z_mm)
                else:
                    factor = self.px_to_mm
                    
                ancho_mm = ancho_px * factor
                
                # Filtro de media móvil
                self.buffer_anchos.append(ancho_mm)
                ancho_mm = float(np.mean(self.buffer_anchos))
                
                # Dibujar en visualización
                box = np.int0(cv2.boxPoints(rect)) * scale + np.array([x1, y1])
                cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
                center = tuple(np.int0(np.array(rect[0]) * scale + np.array([x1, y1])))
                cv2.circle(vis, center, 5, (255, 0, 255), -1)
                cv2.putText(vis, f"{ancho_mm:.2f} mm", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                           
                self.consecutive_failures = 0
                estado_critico = False
            else:
                self.consecutive_failures += 1
                estado_critico = self.consecutive_failures >= self.FAILURE_THRESHOLD
                
            return ancho_mm, vis, mask_binaria, estado_critico
            
        except Exception as e:
            print(f"[MEDIDOR] Error: {e}")
            self.consecutive_failures += 1
            return None, None, None, self.consecutive_failures >= self.FAILURE_THRESHOLD

# ==================== SISTEMA PRINCIPAL ====================

class Sistema3DCP:
    def __init__(self):
        print("\n" + "="*70)
        print("  SISTEMA 3DCP - JETSON NANO PRODUCCIÓN")
        print("="*70 + "\n")
        
        self.safety = SafetySystem()
        self.health = HealthMonitor(self.safety)
        self.tof = DualToFSensor(HW.TOF1_BUS, HW.TOF2_BUS, HW.TOF_ADDR, HW.TOF_TARGET_MM)
        self.controller = None
        self.medidor = None
        
        # Cámaras
        self.pipeline0 = None
        self.pipeline1 = None
        self._stop_cams = threading.Event()
        self.frame_data = {'0': {}, '1': {}}
        self._locks = {'0': threading.Lock(), '1': threading.Lock()}
        
        self.log_file = None
        
    def _init_logger(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG.LOG_DIR, f"sesion_{ts}.csv")
        os.makedirs(LOG.LOG_DIR, exist_ok=True)
        
        with open(log_path, 'w') as f:
            f.write("# SESION 3DCP - " + ts + "\n")
            f.write(",".join(LOG.CSV_COLUMNS) + "\n")
            
        self.log_file = open(log_path, 'a')
        print(f"[LOG] {log_path}")
        
    def log(self, datos):
        if self.log_file:
            fila = [str(datos.get(c, '')) for c in LOG.CSV_COLUMNS]
            self.log_file.write(",".join(fila) + "\n")
            if datos.get('frame', 0) % LOG.FLUSH_INTERVAL == 0:
                self.log_file.flush()
                
    def inicializar(self):
        """Inicialización de hardware"""
        self.safety.start()
        self._init_logger()
        
        if not ORBBEC_AVAILABLE:
            print("[ERROR] Orbbec SDK no disponible")
            return False
            
        # Detectar cámaras
        ctx = Context()
        devices = ctx.query_devices()
        n_cams = devices.get_count()
        print(f"[INIT] Cámaras: {n_cams}")
        
        if n_cams == 0:
            return False
            
        # Configurar cámaras
        self.pipeline0 = Pipeline(devices.get_device_by_index(0))
        self._config_stream(self.pipeline0, "Cam0")
        
        if n_cams > 1:
            self.pipeline1 = Pipeline(devices.get_device_by_index(1))
            self._config_stream(self.pipeline1, "Cam1")
            
        # Iniciar ToF
        self.tof.start()
        
        return True
        
    def _config_stream(self, pipeline, nombre):
        config = Config()
        try:
            perfiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            for w, h, fps in [(640, 480, 30), (640, 480, 15)]:
                try:
                    p = perfiles.get_video_stream_profile(w, h, OBFormat.MJPG, fps)
                    config.enable_stream(p)
                    print(f"[CAM] {nombre}: {w}x{h}@{fps}")
                    break
                except:
                    continue
            pipeline.start(config)
        except Exception as e:
            print(f"[CAM] Error {nombre}: {e}")
            
    def calibrar(self):
        """Calibración interactiva"""
        if not self.safety.set_estado(EstadoSistema.CALIBRANDO):
            return False
            
        print("\n--- CONFIGURACIÓN ---")
        
        # Usar valores por defecto o input
        try:
            altura = input(f"Altura cámara→boquilla [mm] ({HW.ALTURA_REF_MM}): ").strip()
            altura = float(altura) if altura else HW.ALTURA_REF_MM
            
            ancho_obj = float(input("Ancho objetivo [mm]: "))
            vel_base = float(input("Velocidad base [mm/min]: "))
            kp = float(input(f"Kp ({CTRL.KP}): ") or CTRL.KP)
            ki = float(input(f"Ki ({CTRL.KI}): ") or CTRL.KI)
            kd = float(input(f"Kd ({CTRL.KD}): ") or CTRL.KD)
        except ValueError:
            print("Entrada inválida, usando defaults")
            altura = HW.ALTURA_REF_MM
            ancho_obj = 40.0
            vel_base = 1500.0
            kp, ki, kd = CTRL.KP, CTRL.KI, CTRL.KD
            
        # Calcular factor px->mm
        px_to_mm = HW.PX_TO_MM_REF * (altura / HW.ALTURA_REF_MM)
        print(f"\n[CAL] Factor px→mm: {px_to_mm:.6f}")
        
        # Inicializar componentes
        self.medidor = MedidorAnchoFilamento(px_to_mm)
        self.controller = ConcretePIDController(vel_base, ancho_obj, kp, ki, kd)
        
        # Calibrar ToF si hay patrón
        print("\n¿Calibrar ToF con patrón? (s/n): ", end="")
        if input().lower() == 's':
            print("Coloque patrón y presione Enter...")
            input()
            self.tof.calibrar_offset(altura, 30)
            
        print(f"\nResumen: H={altura}mm, Wobj={ancho_obj}mm, Vbase={vel_base}mm/min")
        input("Presione Enter para continuar...")
        
        return True
        
    def warmup(self):
        """Warmup con cámaras activas"""
        if not self.safety.set_estado(EstadoSistema.WARMUP):
            return False
            
        # Iniciar threads de cámara
        self._start_camera_threads()
        
        print(f"\n--- WARMUP ({HW.WARMUP_FRAMES} frames) ---")
        
        for i in range(HW.WARMUP_FRAMES):
            self.safety.ping()
            
            # Leer estado
            tof_st = self.tof.get_state()
            with self._locks['0']:
                a0 = self.frame_data['0'].get('ancho')
            with self._locks['1']:
                a1 = self.frame_data['1'].get('ancho')
                
            self.health.update_tof(tof_st['z1_filtered'], tof_st['z2_filtered'], tof_st['z_fusionado'])
            
            if i % 30 == 0:
                print(f"  {i}/{HW.WARMUP_FRAMES} | ToF: {tof_st['z_fusionado']:.1f}mm | "
                      f"Ancho: {a0 if a0 else 'N/A'}")
                      
            time.sleep(0.02)
            
        print("✓ Warmup OK")
        return True
        
    def _start_camera_threads(self):
        def worker(pipeline, cam_id, lock, data_dict):
            medidor_local = MedidorAnchoFilamento(self.medidor.px_to_mm) if cam_id != '0' else self.medidor
            
            while not self._stop_cams.is_set():
                try:
                    frameset = pipeline.wait_for_frames(100)
                    if not frameset:
                        continue
                        
                    color = frameset.get_color_frame()
                    if not color:
                        continue
                        
                    # Obtener altura ToF actual
                    tof_st = self.tof.get_state()
                    z = tof_st['z_fusionado']
                    
                    # Procesar
                    ancho, vis, mask, critico = medidor_local.procesar(color, z)
                    
                    with lock:
                        data_dict['ancho'] = ancho
                        data_dict['img'] = vis
                        data_dict['mask'] = mask
                        data_dict['ts'] = time.time()
                        data_dict['critico'] = critico
                        
                except Exception as e:
                    time.sleep(0.01)
                    
        self.cam_thread0 = threading.Thread(target=worker, 
                                           args=(self.pipeline0, '0', self._locks['0'], self.frame_data['0']),
                                           daemon=True)
        self.cam_thread0.start()
        
        if self.pipeline1:
            self.cam_thread1 = threading.Thread(target=worker,
                                               args=(self.pipeline1, '1', self._locks['1'], self.frame_data['1']),
                                               daemon=True)
            self.cam_thread1.start()
            
    def run(self):
        """Loop principal de control"""
        if not self.safety.set_estado(EstadoSistema.CONTROL_ACTIVO):
            return
            
        print("\n--- CONTROL ACTIVO ---")
        print("q=quit, e=emergencia, p=pause\n")
        
        frame = 0
        t0 = time.time()
        last_status = t0
        
        # Configurar ventanas
        cv2.namedWindow("Sistema 3DCP", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sistema 3DCP", 1280, 480)
        
        try:
            while not self.safety.is_emergency():
                loop_start = time.time()
                self.safety.ping()
                
                # Leer sensores
                tof_st = self.tof.get_state()
                z = tof_st['z_fusionado']
                
                with self._locks['0']:
                    ancho0 = self.frame_data['0'].get('ancho')
                    img0 = self.frame_data['0'].get('img')
                    mask0 = self.frame_data['0'].get('mask')
                    ts0 = self.frame_data['0'].get('ts', 0)
                    
                # Health
                elapsed = loop_start - t0
                fps = frame / elapsed if elapsed > 0 else 0
                self.health.update_camera(0, ts0, ancho0, fps)
                self.health.update_tof(tof_st['z1_filtered'], tof_st['z2_filtered'], z)
                
                # Verificar operabilidad
                puede, modo = self.health.can_operate()
                if not puede:
                    self.safety.trigger_emergency(f"No operable: {modo}")
                    break
                    
                # Control PID
                gcode, ctrl_dat = self.controller.compute(ancho0, loop_start)
                
                self.health.update_control(ctrl_dat.get('error_mm', 0),
                                          ctrl_dat['velocidad_cmd'],
                                          ctrl_dat['saturado'],
                                          ctrl_dat['deadband'])
                
                # Logging
                frame += 1
                self.log({
                    'timestamp': datetime.now().isoformat(),
                    'estado_sistema': self.safety.get_estado().name,
                    'frame': frame,
                    'ancho_medido_mm': ancho0,
                    'ancho_objetivo_mm': self.controller.ancho_objetivo,
                    'error_mm': ctrl_dat.get('error_mm'),
                    'altura_tof_mm': z,
                    'velocidad_cmd_mmmin': ctrl_dat['velocidad_cmd'],
                    'feedrate_pct': ctrl_dat['feedrate_pct'],
                    'gcode': gcode,
                    'fps': fps,
                    'modo_operacion': modo
                })
                
                # Visualización
                if img0 is not None:
                    # Crear panel de info
                    info = np.zeros((480, 320, 3), dtype=np.uint8)
                    y = 30
                    cv2.putText(info, "CONTROL 3DCP", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y += 35
                    cv2.putText(info, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y += 30
                    cv2.putText(info, f"ToF: {z:.1f}mm" if z else "ToF: N/A", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y += 30
                    cv2.putText(info, f"Ancho: {ancho0:.2f}mm" if ancho0 else "Ancho: N/A", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y += 30
                    cv2.putText(info, f"Error: {ctrl_dat.get('error_mm', 0):+.2f}mm", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if abs(ctrl_dat.get('error_mm', 0)) < 1 else (0, 0, 255), 2)
                    y += 35
                    cv2.putText(info, f"Vel: {ctrl_dat['velocidad_cmd']:.1f}mm/min", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
                    y += 30
                    cv2.putText(info, f"Feed: {ctrl_dat['feedrate_pct']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
                    y += 35
                    cv2.putText(info, f"G-code: {gcode}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y += 35
                    cv2.putText(info, f"Modo: {modo}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Redimensionar imagen cámara
                    if img0.shape[1] != 640:
                        img0 = cv2.resize(img0, (640, 480))
                        
                    # Combinar
                    combined = np.hstack([img0, info])
                    cv2.imshow("Sistema 3DCP", combined)
                    
                # Teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    self.safety.trigger_emergency("Usuario")
                    break
                    
                # Status periódico
                if loop_start - last_status > 2.0:
                    health = self.health.get_system_health()
                    print(f"\n[STATUS] FPS:{fps:.1f} ToF:{z:.1f}mm Ancho:{ancho0:.2f}mm "
                          f"Vel:{ctrl_dat['velocidad_cmd']:.1f} {gcode}")
                    last_status = loop_start
                    
                # Timing
                dt = time.time() - loop_start
                if dt < 0.02:
                    time.sleep(0.02 - dt)
                    
        finally:
            cv2.destroyAllWindows()
            
    def shutdown(self):
        print("\n[SHUTDOWN]")
        self._stop_cams.set()
        if hasattr(self, 'cam_thread0'):
            self.cam_thread0.join(timeout=1)
        if hasattr(self, 'cam_thread1'):
            self.cam_thread1.join(timeout=1)
        if self.pipeline0:
            self.pipeline0.stop()
        if self.pipeline1:
            self.pipeline1.stop()
        self.tof.stop()
        if self.log_file:
            self.log_file.close()
        self.safety.shutdown()
        print("OK")

def main():
    sistema = Sistema3DCP()
    
    try:
        if not sistema.inicializar():
            return 1
        if not sistema.calibrar():
            return 1
        if not sistema.warmup():
            return 1
        sistema.run()
    except KeyboardInterrupt:
        print("\nInterrumpido")
    finally:
        sistema.shutdown()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())