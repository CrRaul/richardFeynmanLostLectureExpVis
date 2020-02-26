
import numpy as np
import cv2
import copy
from collections import deque
from object import *

H,W = 600,1000
G = 1
pathP = deque(maxlen=100)
paForce = deque(maxlen=100)


def attract(obj1, obj2):
    force = obj1.getPosition() - obj2.getPosition()
    dist = np.sqrt(force[0]**2+force[1]**2)
    force = force/dist  # normalize
    strength = (G * obj1.getMass()*obj2.getMass()/(dist**2))
    force *= strength

    return [force,strength]

def scaleVect(p1,p2,factor):
    Xa = p1[0]* (1+factor)/2 + p2[0]* (1-factor)/2
    Ya = p1[1]* (1+factor)/2 + p2[1] *(1-factor)/2

    Xb = p2[0] *(1+factor)/2 + p1[0] *(1-factor)/2
    Yb = p2[1] *(1+factor)/2 + p1[1] *(1-factor)/2

    return np.array([[Xa,Ya],[Xb,Yb]],dtype='float64')

def rotateMiddle90(p1,p2):
    #find the center
    M = np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2],dtype='float64')
    #originTranslate
    p1 -= M
    p2 -= M
    
    #rotate90
    pp1 = [-p1[1],p1[0]]
    pp2 = [-p2[1],p2[0]]

    pp1 += M
    pp2 += M
    
    return [pp1,pp2]

def main():
    Sun = object(W//2,H//2)
    Planet = object(W//2,H//2+52)

    Sun.setMass(1110)

    Planet.setVelocity(np.array([5,0], dtype='float64'))
    Planet.setMass(117)

    while True:
        img = np.zeros((H,W), np.uint8)

        posS = Sun.getPosition()
        posP = Planet.getPosition()
        pathP.appendleft(copy.deepcopy(posP))

        force = attract(Sun,Planet)
        planetForce = force[0]

        paForce.appendleft(copy.deepcopy(force[1]))

        Planet.applyForce(planetForce)
        Planet.update()

        for j in range(1,len(pathP)):
            cv2.line(img,(int(pathP[j-1][0]),int(pathP[j-1][1])),(int(pathP[j][0]),int(pathP[j][1])),(100,100,100),1)
            #line = scale(paForce[j],pathP[j-1],pathP[j])
            
            #draw forceTangent
            [p1,p2] = scaleVect(pathP[j-1],pathP[j],paForce[j])
            cv2.line(img,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(100,100,100),1)

            print(p1,p2)
            #drawParalel
            [pp1,pp2] = rotateMiddle90(p1,p2)
            cv2.line(img,(int(pp1[0]),int(pp1[1])),(int(pp2[0]),int(pp2[1])),(100,100,100),1)

        cv2.circle(img, (int(posS[0]),int(posS[1])), 5, (255,255,255),5)
        cv2.circle(img, (int(posP[0]),int(posP[1])), 5, (255,255,255),1)


        cv2.imshow('gravity-CrRaul', img)
        ch = cv2.waitKey(1)
        if ch == 27:
            break

def testRot():
    img = np.zeros((600,1000), np.uint8)

    p1 = np.array([612.66568614, 325.16459452],dtype='float64')
    p2 = np.array([412.13676834, 375.17579168],dtype='float64')
    line = rotateMiddle90(p1,p2)

    while True:

        cv2.line(img,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(255,255,255),2)
        cv2.line(img,(int(line[0][0]),int(line[0][1])),(int(line[1][0]),int(line[1][1])),(100,100,100),1)

        cv2.imshow('gravity-CrRaul', img)

        ch = cv2.waitKey(1)
        if ch == 27:
            break

if __name__ == '__main__':
    main()
    #testRot()
    #print(rotateMiddle90(np.array([3,3],dtype='float64'),np.array([-1,-1],dtype='float64')))

cv2.destroyAllWindows()
cv2.waitKey(1)