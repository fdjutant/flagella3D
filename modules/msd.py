import numpy as np
from numba import jit

#%% Namba
@jit(nopython=True)
def trans_stepSize_Namba(cm, n1, n2, n3):
    
    # compute step size from adjacent points
    stepSize0 = []; stepSize1 = []; stepSize2 =[];
    for i in range(len(cm)-1):
        
        # step size in Cartesian coordinates
        deltaX = ( cm[i+1,0] - cm[i,0] ) * 0.115
        deltaY = ( cm[i+1,1] - cm[i,1] ) * 0.115
        deltaZ = ( cm[i+1,2] - cm[i,2] ) * 0.115
        deltaXYZ = np.array([deltaX, deltaY, deltaZ])
        
        # step size in local axes
        stepSize0.append(n1[i,0]*deltaXYZ[0] + 
                         n1[i,1]*deltaXYZ[1] +
                         n1[i,2]*deltaXYZ[2]) # parallel
        stepSize1.append(n2[i,0]*deltaXYZ[0] + 
                         n2[i,1]*deltaXYZ[1] +
                         n2[i,2]*deltaXYZ[2]) # perp1
        stepSize2.append(n3[i,0]*deltaXYZ[0] + 
                         n3[i,1]*deltaXYZ[1] +
                         n3[i,2]*deltaXYZ[2]) # perp2
        
    return stepSize0, stepSize1, stepSize2

@jit(nopython=True)
def rot_stepSize_Namba(EuAng):
        
    # compute step size from adjacent points
    # stepSize = np.zeros([len(EuAng)-1,3])
    stepSize0 = []; stepSize1 = []; stepSize2 =[];
    for i in range(len(EuAng)-1):        
        stepSize0.append(EuAng[i+1,0] - EuAng[i,0])   # pitch
        stepSize1.append(EuAng[i+1,1] - EuAng[i,1])   # roll
        stepSize2.append(EuAng[i+1,2] - EuAng[i,2])   # yaw
        
    return stepSize0, stepSize1, stepSize2

@jit(nopython=True)
def trans_MSD_Namba(Nframes, cm, rollAng, n1, n2, n3, expTime, nInterval):

    # compute translation MSD at 3 different axes
    MSD_S1 = np.zeros(nInterval); MSD_N = np.zeros(nInterval);
    MSD_NR = np.zeros(nInterval); MSD_S2 = np.zeros(nInterval);
    j = 1;
    while j < nInterval+1:
        tempN = []; tempS1 = []; tempS2 = []; tempNR = [];
        i = 0;
        while i + j <= Nframes-1:
            
            temp1 =[]; temp2 = []; temp3 = [];
            k = 0;
            while k < j:
                deltaXYZ = (cm[i+k+1,:]-\
                            cm[i+k,:])*0.115
                temp1.append(n1[i+k,0]*deltaXYZ[0] +
                             n1[i+k,1]*deltaXYZ[1] + 
                             n1[i+k,2]*deltaXYZ[2])
                temp2.append(n2[i+k,0]*deltaXYZ[0] +
                             n2[i+k,1]*deltaXYZ[1] + 
                             n2[i+k,2]*deltaXYZ[2])
                temp3.append(n3[i+k,0]*deltaXYZ[0] +
                             n3[i+k,1]*deltaXYZ[1] + 
                             n3[i+k,2]*deltaXYZ[2])
                k += 1
            
            tempN.append(np.sum(np.array(temp1))**2)
            tempS1.append(np.sum(np.array(temp2))**2)
            tempS2.append(np.sum(np.array(temp3))**2)
            tempR = rollAng[i+j] - rollAng[i]
            
            tempNR.append(np.sum(np.array(temp1))*tempR)
            
            i += 1
        MSD_N[j-1]  = np.mean(np.array(tempN))
        MSD_S1[j-1] = np.mean(np.array(tempS1))
        MSD_S2[j-1] = np.mean(np.array(tempS2))
        MSD_NR[j-1] = np.mean(np.array(tempNR))
        j += 1
    
    return MSD_N, MSD_S1, MSD_S2, MSD_NR

@jit(nopython=True)
def trans_MSD_direct_Namba(Nframes, cm, rollAng, n1, n2, n3, expTime, nInterval):

    # compute translation MSD at 3 different axes
    MSD_S1 = np.zeros(nInterval); MSD_N = np.zeros(nInterval);
    MSD_NR = np.zeros(nInterval); MSD_S2 = np.zeros(nInterval);
    j = 1;
    while j < nInterval+1:
        tempN = []; tempS1 = []; tempS2 = []; tempNR = [];
        i = 0;
        while i + j <= Nframes-1:
            deltaXYZ = (cm[i+j,:]-cm[i,:])*0.115
            tempN.append((n1[i,0]*deltaXYZ[0] +
                          n1[i,1]*deltaXYZ[1] + 
                          n1[i,2]*deltaXYZ[2])**2)
            tempS1.append((n2[i,0]*deltaXYZ[0] +
                           n2[i,1]*deltaXYZ[1] + 
                           n2[i,2]*deltaXYZ[2])**2)
            tempS2.append((n3[i,0]*deltaXYZ[0] +
                           n3[i,1]*deltaXYZ[1] + 
                           n3[i,2]*deltaXYZ[2])**2)
            tempNR.append((n1[i,0]*deltaXYZ[0] + n1[i,1]*deltaXYZ[1] + 
                           n1[i,2]*deltaXYZ[2]) * (rollAng[i+j] - rollAng[i]))
            i += 1
        MSD_N[j-1]  = np.mean(np.array(tempN))
        MSD_S1[j-1] = np.mean(np.array(tempS1))
        MSD_S2[j-1] = np.mean(np.array(tempS2))
        MSD_NR[j-1] = np.mean(np.array(tempNR))
        j += 1
    
    return MSD_N, MSD_S1, MSD_S2, MSD_NR

@jit(nopython=True)
def regMSD_Namba(Nframes, cm, vol_exp, nInterval):
              
    MSD = np.zeros(nInterval)
    j = 1
    while j < nInterval+1:
        temp = []
        i = 0
        while i + j <= Nframes-1:
            temp.append((cm[i+j] - cm[i])**2)
            i += 1
        MSD[j-1] = np.mean(np.array(temp))
        j += 1
        
    return MSD

#%% without Namba
def regMSD(Nframes, cm, vol_exp, nInterval):
              
    MSD = np.zeros(nInterval)
    j = 1
    while j < nInterval+1:
        temp = []
        i = 0
        while i + j <= Nframes-1:
            temp.append((cm[i+j] - cm[i])**2)
            i += 1
        MSD[j-1] = np.mean(temp)
        j += 1
        
    return MSD

def trans_stepSize_all(cm, localAxes):
        
    # import major vector        
    n1 = localAxes[:,0]; n2 = localAxes[:,1];
    n3 = localAxes[:,2]

    # compute step size from adjacent points
    stepSize_all = []
    for j in range(1,len(cm)):

        N = np.round((len(cm)-1)/j).astype('int')
        stepSize = np.zeros([N,3])
        for i in range(N):
            
            # step size in Cartesian coordinates
            deltaX = ( cm[i+1,0] - cm[i,0] ) * 0.115
            deltaY = ( cm[i+1,1] - cm[i,1] ) * 0.115
            deltaZ = ( cm[i+1,2] - cm[i,2] ) * 0.115
            deltaXYZ = np.array([deltaX, deltaY, deltaZ])
            
            # step size in local axes
            stepSize[i,0] = np.dot(n1[i,:],deltaXYZ) # parallel
            stepSize[i,1] = np.dot(n2[i,:],deltaXYZ) # perp1
            stepSize[i,2] = np.dot(n3[i,:],deltaXYZ) # perp2
        
        stepSize_all.append(stepSize)
        
    stepSize_all = np.array(stepSize_all, dtype=object)
    
    return stepSize_all


class theMSD:
    def __init__(self, Nframes, cm, rollAng, localAxes, expTime, nInterval):
        self.Nframes = Nframes
        self.cm = cm
        self.rollAng = rollAng
        self.localAxes = localAxes
        self.expTime = expTime
        self.nInterval = nInterval 
        
    def trans_combo_MSD(self):
        
        # import major vector        
        n1 = self.localAxes[:,0]; n2 = self.localAxes[:,1];
        n3 = self.localAxes[:,2]

        # compute translation MSD at 3 different axes
        MSD_S1 = np.zeros(self.nInterval); MSD_N = np.zeros(self.nInterval);
        MSD_NR = np.zeros(self.nInterval); MSD_S2 = np.zeros(self.nInterval);
        j = 1;
        while j < self.nInterval+1:
            tempN = []; tempS1 = []; tempS2 = []; tempNR = [];
            i = 0;
            while i + j <= self.Nframes-1:
                
                temp1 =[]; temp2 = []; temp3 = [];
                k = 0;
                while k < j:
                    deltaXYZ = (self.cm[i+k+1,:]-\
                                self.cm[i+k,:])*0.115
                    temp1.append(np.dot(n1[i+k,:],deltaXYZ))
                    temp2.append(np.dot(n2[i+k,:],deltaXYZ))
                    temp3.append(np.dot(n3[i+k,:],deltaXYZ))
                    k += 1
                
                tempN.append(np.sum(temp1)**2)
                tempS1.append(np.sum(temp2)**2)
                tempS2.append(np.sum(temp3)**2)
                tempR = self.rollAng[i+j] - self.rollAng[i]
                
                tempNR.append(np.sum(temp1)*tempR)
                
                i += 1
            MSD_N[j-1] = np.mean(tempN)
            MSD_S1[j-1] = np.mean(tempS1)
            MSD_S2[j-1] = np.mean(tempS2)
            MSD_NR[j-1] = np.mean(tempNR)
            j += 1
        
        return MSD_N, MSD_S1, MSD_S2, MSD_NR


class theSS:
    def __init__(self, Nframes, cm, rollAng, localAxes, expTime, nInterval):
        self.Nframes = Nframes
        self.cm = cm
        self.rollAng = rollAng
        self.localAxes = localAxes
        self.expTime = expTime
        self.nInterval = nInterval
        
    def trans_SS(self):
        
        # import major vector        
        n1 = self.localAxes[:,0]; n2 = self.localAxes[:,1];
        n3 = self.localAxes[:,2]

        # compute translation MSD at 3 different axes
        SS_N = []; SS_S1 = []; SS_S2 = [];
        j = 1;
        while j < self.nInterval+1:
            tempN = []; tempS1 = []; tempS2 = []; 
            i = 0;
            while i + j <= self.Nframes-1:
                
                temp1 =[]; temp2 = []; temp3 = [];
                k = 0;
                while k < j:
                    deltaXYZ = (self.cm[i+k+1,:]-\
                                self.cm[i+k,:])*0.115
                    temp1.append(np.dot(n1[i+k,:],deltaXYZ))
                    temp2.append(np.dot(n2[i+k,:],deltaXYZ))
                    temp3.append(np.dot(n3[i+k,:],deltaXYZ))
                    k += 1
                
                tempN.append(np.sum(temp1))
                tempS1.append(np.sum(temp2))
                tempS2.append(np.sum(temp3))
                
                i += 1
            SS_N.append(tempN)
            SS_S1.append(tempS1)
            SS_S2.append(tempS2)
            j += 1
        SS_N = np.array(SS_N, dtype=object)
        SS_S1 = np.array(SS_S1, dtype=object)
        SS_S2 = np.array(SS_S2, dtype=object)
        
        return SS_N, SS_S1, SS_S2

def trans_stepSize(cm, localAxes):
        
    # import major vector        
    n1 = localAxes[:,0]; n2 = localAxes[:,1];
    n3 = localAxes[:,2]

    # compute step size from adjacent points
    stepSize = np.zeros([len(cm)-1,3])
    for i in range(len(cm)-1):
        
        # step size in Cartesian coordinates
        deltaX = ( cm[i+1,0] - cm[i,0] ) * 0.115
        deltaY = ( cm[i+1,1] - cm[i,1] ) * 0.115
        deltaZ = ( cm[i+1,2] - cm[i,2] ) * 0.115
        deltaXYZ = np.array([deltaX, deltaY, deltaZ])
        
        # step size in local axes
        stepSize[:,0] = np.dot(n1[i,:],deltaXYZ) # parallel
        stepSize[:,1] = np.dot(n2[i,:],deltaXYZ) # perp1
        stepSize[:,2] = np.dot(n3[i,:],deltaXYZ) # perp2
        
    return stepSize

def rot_stepSize(EuAng):
        
    # compute step size from adjacent points
    stepSize = np.zeros([len(EuAng)-1,3])
    for i in range(len(EuAng)-1):        
        stepSize[i,0] = EuAng[i+1,0] - EuAng[i,0]   # pitch
        stepSize[i,1] = EuAng[i+1,1] - EuAng[i,1]   # roll
        stepSize[i,2] = EuAng[i+1,2] - EuAng[i,2]   # yaw
        
    return stepSize
