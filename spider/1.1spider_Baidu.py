import urllib.request
response=urllib.request.urlopen('http://www.baidu.com')
print(response.read().decode('utf-8'))
#  结果  被目标计算机拒绝，无法连接 ？