# Generates PNG fallback assets for README
import base64
from pathlib import Path

# 1x1 transparent PNG (base64)
icon_b64 = b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAqMB0ZqJq+oAAAAASUVORK5CYII='
# Let's reuse same for banner (small fallback)
banner_b64 = icon_b64

assets_dir = Path(__file__).parent
icon_path = assets_dir / 'icon.png'
banner_path = assets_dir / 'banner.png'

icon_path.write_bytes(base64.b64decode(icon_b64))
banner_path.write_bytes(base64.b64decode(banner_b64))

print(f'Wrote {icon_path} and {banner_path}')
