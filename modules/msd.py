from matmatrix import rotmat, movAverage, phaseUnwrap
import numpy as np
from scipy import stats

def regMSD(Nframes, cm, vol_exp, nInterval):
              
    # create x-axis for plotting
    time_x = np.linspace(0,nInterval,nInterval)*vol_exp  
    MSD = np.zeros(nInterval)
    
    j = 1;
    while j < nInterval:
        temp =[];
        i = 0;
        while i + j <= Nframes-1:
            temp.append((cm[i+j] - cm[i])**2)
            i += 1
        MSD[j] = np.mean(temp)
        j += 1
        
    return time_x, MSD

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
        stepSize[i,0] = np.dot(n1[i,:],deltaXYZ) # parallel
        stepSize[i,1] = np.dot(n2[i,:],deltaXYZ) # perp1
        stepSize[i,2] = np.dot(n3[i,:],deltaXYZ) # perp2
        
    return stepSize


def rot_stepSize(EuAng):
        
    # compute step size from adjacent points
    stepSize = np.zeros([len(EuAng)-1,3])
    for i in range(len(EuAng)-1):        
        stepSize[i,0] = ( EuAng[i+1,0] - EuAng[i,0] )   # pitch
        stepSize[i,1] = ( EuAng[i+1,1] - EuAng[i,1] )   # roll
        stepSize[i,2] = ( EuAng[i+1,2] - EuAng[i,2] )   # yaw
        
    return stepSize

class theMSD:
    def __init__(self, Nframes, cm, rollAng, localAxes, expTime, nInterval):
        self.Nframes = Nframes
        self.cm = cm
        self.rollAng = rollAng
        self.localAxes = localAxes
        self.expTime = expTime
        self.nInterval = nInterval
        
    def time_MSD(self):
        
        # compute exposure time
        vol_exp = self.expTime
               
        # create x-axis for plotting
        time_x = np.linspace(0,self.nInterval,self.nInterval)*vol_exp  
    
        return time_x  
        
    def trans_combo_MSD(self):

        time_x = self.time_MSD()
        
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
        
        return time_x, MSD_N, MSD_S1, MSD_S2, MSD_NR
    
    

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
    