# -*- coding: utf-8 -*-
"""把模板 + three.min.js + 数据 合成为单文件 HTML（离线可用、双击即开）"""
import json, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
K6 = r"D:\k6data"


def resolve(name):
    """优先用本目录下的同名文件，退化到 D:\\k6data（构建时的实际位置）"""
    local = os.path.join(HERE, name)
    return local if os.path.isfile(local) else os.path.join(K6, name)


TPL = resolve("viewer_template.html")
THREE = resolve("three.min.js")          # three.js r128 UMD（unpkg three@0.128.0）
DATA = resolve("viewer_data.json")       # 由 export_web.py 生成
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "port_viewer.html")

for _f in (TPL, THREE, DATA):
    if not os.path.isfile(_f):
        raise SystemExit("!! 缺少依赖文件：%s" % _f)

tpl = io.open(TPL, encoding="utf-8").read()
three = io.open(THREE, encoding="utf-8").read()
data = io.open(DATA, encoding="utf-8").read()

# 安全性检查：内嵌脚本里出现 </script> 会提前闭合标签
for nm, s in (("three.min.js", three), ("data", data)):
    if "</script" in s.lower():
        raise SystemExit("!! %s 含 </script>，需转义" % nm)

html = tpl.replace("/*__THREE__*/", three)
html = html.replace("/*__DATA__*/", "window.VIEW_DATA = " + data + ";")

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

print("已生成 %s  (%.2f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
