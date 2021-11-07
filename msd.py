from matmatrix import rotmat, movAverage, phaseUnwrap
import numpy as np
from scipy import stats

def regMSD(rNlagTime, Nframes, cm, vol_exp):
           
    # decide how long the lag time will be
    NlagTime = np.round(rNlagTime*Nframes,0).astype(int);
    
    # create x-axis for plotting
    time_x = np.linspace(0,NlagTime,NlagTime)*vol_exp  
    MSD = np.zeros(NlagTime)
    
    j = 1;
    while j < NlagTime:
        temp =[]; temp2 =[];
        i = 0;
        while i + j <= Nframes-1:
            temp.append((cm[i+j] - cm[i])**2)
            i += 1
        MSD[j] = np.mean(temp)
        j += 1
        
    return time_x, MSD

class theMSD:
    def __init__(self, rNlagTime, Nframes, cm, dirAng,\
                 rollAng, localAxes, expTime):
        self.rNlagTime = rNlagTime
        self.Nframes = Nframes
        self.dirAng = dirAng
        self.cm = cm
        self.rollAng = rollAng
        self.localAxes = localAxes
        self.expTime = expTime
        
    def time_MSD(self):
        
        # compute exposure time
        vol_exp = self.expTime
        
        # decide how long the lag time will be
        NlagTime = np.round(self.rNlagTime*self.Nframes,0).astype(int);
        
        # create x-axis for plotting
        time_x = np.linspace(0,NlagTime,NlagTime)*vol_exp  
    
        return NlagTime, time_x

    def trans_combo_MSD(self):

        NlagTime, time_x = self.time_MSD()
        
        # import major vector        
        n1 = self.localAxes[:,0]

        # compute translation MSD at 3 different axes
        MSD_S = np.zeros(NlagTime); MSD_N = np.zeros(NlagTime);
        MSD_NR = np.zeros(NlagTime);
        j = 1;
        while j < NlagTime:
            tempN = []; tempS = []; tempNR = [];
            i = 0; 
            while i + j <= self.Nframes-1:
                temp1 =[]; temp2 = [];
                
                deltaX = ( self.cm[i+j,0] - self.cm[i,0] ) * 0.115
                deltaY = ( self.cm[i+j,1] - self.cm[i,1] ) * 0.115
                deltaZ = ( self.cm[i+j,2] - self.cm[i,2] ) * 0.115
                
                temp1 = 0.5 * (n1[i,0] + n1[i+j,0]) * deltaX +\
                        0.5 * (n1[i,1] + n1[i+j,1]) * deltaY +\
                        0.5 * (n1[i,2] + n1[i+j,2]) * deltaZ
                # temp1 = n1[i,0] * deltaX +\
                #         n1[i,1] * deltaY +\
                #         n1[i,2] * deltaZ
                temp2 = np.linalg.norm([ (deltaX - temp1*n1[i,0]),\
                                         (deltaY - temp1*n1[i,1]),\
                                         (deltaZ - temp1*n1[i,2]) ])                
                        
                tempN.append(temp1**2)
                tempS.append(temp2**2)
                tempR = self.rollAng[i+j] - self.rollAng[i]
                
                tempNR.append(temp1*tempR)
                
                i += 1
            MSD_N[j] = np.mean(tempN)
            MSD_S[j] = np.mean(tempS)
            MSD_NR[j] = np.mean(tempNR)
            j += 1
        
        return time_x, MSD_N, MSD_S, MSD_NR
    
'''
    def trans_MSD_old(self):
    
        NlagTime, time_x = self.time_MSD()
        
        # compute moving average
        P = np.zeros(NlagTime); R = np.zeros(NlagTime); Y = np.zeros(NlagTime)
        j = 1
        while j < NlagTime:
            P[j] = np.mean(movAverage(abs(self.dirAng[:,0]), j))
            R[j] = np.mean(movAverage(abs(self.dirAng[:,1]), j))
            Y[j] = np.mean(movAverage(abs(self.dirAng[:,2]), j))
            j += 1
        
        # compute translation MSD at 3 different axes
        MSD_S1 = np.zeros(NlagTime); MSD_S2 = np.zeros(NlagTime)
        MSD_N = np.zeros(NlagTime);  MSD_NR = np.zeros(NlagTime) 
        j = 1;
        while j < NlagTime:
            tempN = []; tempS1 =[]; tempS2 = []; tempNR = [];
            i = 0; T = [];
            T = rotmat(np.array([self.localAxes[j,0],\
                                  self.localAxes[j,1],\
                                  self.localAxes[j,2]]))
            # T = rotmat(np.array([P[j],R[j],Y[j]])) # rot matrix for time-averaging
            while i + j <= self.Nframes-1:
                temp =[]
                
                deltaX = self.cm[i+j,0] - self.cm[i,0]
                deltaY = self.cm[i+j,1] - self.cm[i,1]
                deltaZ = self.cm[i+j,2] - self.cm[i,2]
                deltaXYZ = np.array([deltaX,deltaY,deltaZ])*0.115
                
                temp = np.matmul(T,deltaXYZ)        
                tempN.append(temp[0]**2)
                tempS1.append(temp[1]**2)
                tempS2.append(temp[2]**2)
                
                tempR = self.rollAng[i+j] - self.rollAng[i]
                
                tempNR.append(temp[0]*tempR)
                
                i += 1
            MSD_N[j] = np.mean(tempN)
            MSD_S1[j] = np.mean(tempS1)
            MSD_S2[j] = np.mean(tempS2)
            MSD_NR[j] = np.mean(tempNR)
            j += 1
            
        return time_x, MSD_N, MSD_S1, MSD_NR
       
    def combo_MSD_old(self):
                
        # Create the time lag axis
        NlagTime, time_x = self.time_MSD()
        
        # Compute moving average
        P = np.zeros(NlagTime); R = np.zeros(NlagTime); Y = np.zeros(NlagTime)
        j = 1
        while j < NlagTime:
            P[j] = np.mean(movAverage(abs(self.dirAng[:,0]), j))
            R[j] = np.mean(movAverage(abs(self.dirAng[:,1]), j))
            Y[j] = np.mean(movAverage(abs(self.dirAng[:,2]), j))
            j += 1
        
        # compute MSD combo
        MSD_NR = np.zeros(NlagTime) 
        j = 1;
        while j < NlagTime:
            tempNR = [];
            i = 0; T = [];
            T = rotmat(np.array([P[j],R[j],Y[j]])) # rot matrix for time-averaging
            while i + j <= self.Nframes-1:
                tempN = []; tempR = [];
                
                deltaX = self.cm[i+j,0] - self.cm[i,0]
                deltaY = self.cm[i+j,1] - self.cm[i,1]
                deltaZ = self.cm[i+j,2] - self.cm[i,2]
                deltaXYZ = np.array([deltaX,deltaY,deltaZ])*0.115
                
                tempN = np.matmul(T,deltaXYZ) # deltaN, deltaS1, deltaS2
                tempR = self.rollAng[i+j] - self.rollAng[i]
                
                tempNR.append(tempN[1]*tempR)
                
                i += 1
            MSD_NR[j] = np.mean(tempNR)
            j += 1
            
        return time_x, MSD_NR
'''