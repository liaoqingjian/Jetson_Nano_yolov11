import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

class RTSPServer:
    def __init__(self):
        Gst.init(None)

        self.server = GstRtspServer.RTSPServer()
        self.server.set_service('8554')
        self.factory = GstRtspServer.RTSPMediaFactory()
        launch_string = f'nvarguscamerasrc ' \
                        f'! video/x-raw(memory:NVMM), width=1280, height=720, ' \
                        f'framerate=30/1 ! nvv4l2h264enc ! rtph264pay name=pay0 pt=96'
        self.factory.set_launch(launch_string)
        #self.factory.set_launch('( nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=I420 ! omxh264enc ! rtph264pay name=pay0 pt=96 )')
        self.factory.set_shared(True)

        self.mounts = self.server.get_mount_points()
        self.mounts.add_factory('/stream', self.factory)

        self.server.attach(None)
        print("RTSP stream ready at rtsp://localhost:8554/stream")

if __name__ == "__main__":
    server = RTSPServer()
    loop = GLib.MainLoop()
    loop.run()