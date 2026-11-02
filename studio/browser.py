"""Keep upload available in browsers that do not expose microphone APIs."""

HEAD = '''<script>
if (!navigator.mediaDevices) {
 Object.defineProperty(navigator, 'mediaDevices', {configurable: true, value: {
   enumerateDevices: async () => [],
   getUserMedia: async () => {
     throw new DOMException('Microphone recording needs HTTPS and browser support. You can upload a recording.', 'NotSupportedError');
   }
 }});
}
</script>'''
