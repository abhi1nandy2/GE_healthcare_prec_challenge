import os
prev_time=os.stat('doc.').st_mtime
while(1):
	if(os.stat('nandu.py').st_mtime!=prev_time):
		print('hello')
		prev_time = os.stat('nandu.py').st_mtime