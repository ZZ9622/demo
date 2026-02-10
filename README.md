＃＃＃ Instruction

Back up data source: https://humansensinglab.github.io/basket-multiview/data.html

After downloading, you can do like this:
'''
ffmpeg -framerate 25 -i %04d.png -c:v libx264 -pix_fmt yuv420p xxx.mp4
'''