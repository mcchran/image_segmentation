'''
	The factory fabricating thread safe genrators.
	The way that they should
'''

import threading

class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return self.it.__next__()

def threadsafe_generator(f):
	'''
		Decorator that returns the thread safe iterator
	'''
	def g(*a, **kw): # the stars indeicate the arbitraty number of nonamed and named params
		return threadsafe_iter(f(*a, **kw))
	return g

