
import numpy as np

class object():
	def __init__(self, x,y):
		self.__position = np.array([x,y], dtype='float64')
		self.__velocity = np.array([0,0], dtype='float64')
		self.__acceleration = np.array([0,0], dtype='float64')
		self.__mass = 0

	def setPosition(self,position):
		self.__position = position
	def setVelocity(self,velocity):
		self.__velocity = velocity
	def setAcceleration(self,acceleration):
		self.__acceleration = acceleration
	def setMass(self,mass):
		self.__mass = mass

	def getPosition(self):
		return self.__position
	def getVelocity(self):
		return self.__velocity
	def getAcceleration(self):
		return self.__acceleration
	def getMass(self):
		return self.__mass


	def applyForce(self, force):
		f = force / self.__mass
		self.__acceleration = self.__acceleration+f

	def update(self):
		self.__velocity += self.__acceleration
		self.__position += self.__velocity
		self.__acceleration *= 0
