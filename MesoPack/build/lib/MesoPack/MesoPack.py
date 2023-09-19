# python3
##################################
from __future__ import division
import numpy as np
from numpy import dtype, linalg
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as sclin
import itertools as it
import h5py
import os
import shutil
import pickle
import timeit
import copy
##################################

##################################
# global quantities 
num_bas = 4
##################################

def fermi_parent(T):
    # fermi dirac distribution
    def fermi(x):
        return 1./(1+np.exp(x/T))
    return fermi

def totspin(s):
    # Turn total spin S(S+1) value into S value
    return int(-1+np.sqrt(num_bas*s + 1))

def product(*args,bcset, repeat=1):
    # itertools product function, with charge boundary condition
    # enforcement build in
    # bcset = [[N_low,N_high], [site_1,stie_2]], site_1 and site_2 are the boundary sites
    # and can assume the same value.
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools[:-1]:
            result = [x+[y] for x in result for y in pool]
    pool = pools[-1]
    result = [x+[y] for x in result for y in pool\
                   if sum(bcset[0][0]<=np.abs((x+[y])[bcset[1][0]:bcset[1][1]+1]))<=bcset[0][1]]
    for prod in result:
        yield tuple(prod)

def state2num(state):
    return int(sum([ (s+1)*(4**i) for i,s in enumerate(state)]))

##########################################
class OperatorBase(object):
    def __init__(self, H, opA, deltaQN): 
        self.H = H
        self.opA = opA
        self.deltaQN = deltaQN
        self.qnCount = H.qnCount
        
    def QnInd(self,qn_string):
        return self.H.qnind[qn_string]
        
    def GetMat(self,qnind,sparse=True):
        #return operator
        if qnind >= self.qnCount:
            print("QN index out of range")
            return None
        if sparse:
            return self.opA[qnind]
        else:
            return self.opA[qnind].toarray()
        
    def OperatorSum(self,opaB,factor=1.):
        # sum operators
        if not opaB.deltaQN == self.deltaQN:
            print("Operators must have same change of QN for summation")
            return None
        # get max qunatum numbers
        max_q = self.H.max_q
        max_sz = self.H.max_sz
        # We go through all the interactions to create the Hamiltonian 
        for q in np.arange(-max_q,max_q+1):
            for sz in np.arange(-max_sz,max_sz+1):

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): 
                    continue
                
                # getting the index of the ket QN
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                
                # sum the operators
                self.opA[qnind] += factor*opaB.GetMat(qnind)
        print("Operators were summed")
        return None
                
    def transpose(self):
        # transpose the operator
        # new delta QN
        self.deltaQN[0] *= -1
        self.deltaQN[1] *= -1
        # reverse QNs
        rev_dict = {}
        for qnind,QNs in enumerate(self.H.qnind):
            QNint = [int(x) for x in QNs[1::].split("S",1)]
            QNstring = "Q%.iS%.i"%(QNint[0]+self.deltaQN[0],QNint[1]+self.deltaQN[1])
            if QNstring in self.H.qnind: 
                new_ind = self.H.qnind[QNstring]
                rev_dict[new_ind] = qnind
            else:
                QNstring = "Q%.iS%.i"%(-1.*(QNint[0]),-1.*(QNint[1]))
                new_ind = self.H.qnind[QNstring]
                rev_dict[new_ind] = qnind
        # loop over QNs
        opaT = self.opA.copy()
        for qnind,QNs in enumerate(self.H.qnind):
            if qnind in rev_dict:
                new_ind = rev_dict[qnind]
                opaT[new_ind] = self.opA[qnind].T
        self.opA = opaT
        return None
    
    def OperatorDot(self,opaB,factor=1.):
        # multiply the current operator to opB
        max_q = self.H.max_q
        max_sz = self.H.max_sz
        # We go through all the interactions to create the Hamiltonian 
        for q in np.arange(-max_q,max_q+1):
            for sz in np.arange(-max_sz,max_sz+1):

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): 
                    continue

                # getting the index of the ket QN
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q+self.deltaQN[0],sz+self.deltaQN[1]) in self.H.qnind): 
                    continue

                # getting the index of the bra QN
                qnind_new = self.H.qnind["Q%.iS%.i"%(q+self.deltaQN[0],sz+self.deltaQN[1])] 
                
                # multiply the operators
                self.opA[qnind] = factor*sparse.csr_matrix(np.matmul(opaB.GetMat(qnind_new,False),self.GetMat(qnind,False)))
        # adjust the quantum numbers
        self.deltaQN[0] += opaB.deltaQN[0]
        self.deltaQN[1] += opaB.deltaQN[1]
        print("new change in quantum numbers: ",self.deltaQN)
        return None
     
    def KetDot(self,ket):
        # multiply operator with ket
        QNs, kets = ket
        QNint = [int(x) for x in QNs[1::].split("S",1)]
        qnind = self.H.qnind[QNs]
        NewQN = [QNint[0] + self.deltaQN[0]]
        NewQN += [QNint[1] + self.deltaQN[1]]
        if not "Q%.iS%.i"%(NewQN[0],NewQN[1]) in self.H.qnind:
            #print("Quantum number mistmatch")
            #print(ketQNint,"=/=",braQNint)
            return (None,0.)
        NewKet = np.matmul(self.opA[qnind].toarray(),kets)
        return ("Q%.iS%.i"%(NewQN[0],NewQN[1]),NewKet)
    
    def ScalarDot(self,V):
        # multiply operator with ket
        self.opA = [V*op for op in self.opA]
        return None
    
    def BraDot(self,bra):
         # multiply operator with bra
        print("Not implemented yet")
        return None
    
    def BraKetDot(self,bra,ket):
        # scalar product
        # multiply operator with ket
        ketQNs, kets = ket
        ketQNint = [int(x) for x in ketQNs[1::].split("S",1)]
        braQNs, bras = bra
        braQNint = [int(x) for x in braQNs[1::].split("S",1)]
        qnind = self.H.qnind[ketQNs]
        ### new QN
        ketQNint[0] += self.deltaQN[0]
        ketQNint[1] += self.deltaQN[1]
        if not "Q%.iS%.i"%(ketQNint[0],ketQNint[1]) == braQNs:
            #print("Quantum number mistmatch")
            #print(ketQNint,"=/=",braQNint)
            return (None,0.)
        NewKet = np.matmul(self.opA[qnind].toarray(),kets)
        ScalarProd = np.matmul(np.conjugate(bras.T),NewKet)
        # use vectorzied dot product to multiply NewKet and bra
        return (braQNs,ScalarProd)
    
    def Diagonalize(self,sparse=False,Relative2GS=True,Sort=True,verbosity=2):
        # diagonalize by quantum number subsapce
        start = timeit.default_timer()
        self.states = []
        self.energies = []
        eg = []
        if not self.deltaQN == [0,0]:
            return self.energies, self.states
        for i,(qn,h) in enumerate(zip(self.H.qnind,self.opA)):
            if verbosity>1: print(qn,f'{"> start":->10}')
            if h.nnz >= h.shape[0] or sparse == False or h.shape[0] < 6:
                v, w = np.linalg.eigh(h.toarray())
                self.energies += [(qn,v)]
                self.states += [(qn,w.T)]
                eg += [(v[0],i)]
            else:
                if verbosity>1: print("Matrix density: %.2f "%(100.*(h.nnz/h.shape[0]**2)),"%")
                v, w = sclin.eigsh(h)
                self.energies += [(qn,v)]
                self.states += [(qn,w.T)]
                eg += [(v[0],i)]
            if verbosity>1: print(f'{"> done":->25}')
        if Sort:
            # sorting ground state energies
            eg = sorted(eg)
            # sorting energies and state
            if Relative2GS:
                self.energies = [(en[0],en[1]-eg[0][0]) for en in self.energies]
                self.energies = [self.energies[x[1]] for x in eg]
            else:
                self.energies = [self.energies[x[1]] for x in eg]
            self.states = [self.states[x[1]] for x in eg]
            stop = timeit.default_timer()
            m, s = divmod(stop-start, 60)
            h, m = divmod(m, 60)
            if verbosity>=1: print('Runtime: %.0d:%.2d:%.2d'%(h, m, s))
            return self.energies, self.states
        else:
            stop = timeit.default_timer()
            m, s = divmod(stop-start, 60)
            h, m = divmod(m, 60)
            if verbosity>=1: print('Runtime: %.0d:%.2d:%.2d'%(h, m, s))
            return self.energies, self.states
            
    
    def trace(self):
        # trace of operator
        return sum([np.trace(op.toarray()) for op in self.opA])
    
    def copy(self):
        # create an independent copy of the present class
        return copy.deepcopy(self)

##########################################
# functions operating on operater objects
##########################################
def SuperBlockDiagonalization(SymmetrySet,hamil,verbosity=False,BlockOperator=True,SuperTrafo=[]):
    ###############################
    # Mutually diagonalizes an arbitrary number of non-abelian symmetries and and a given Hamiltonian
    # SymmetrySet: List of mutually commuting operator objects
    # hamil: the Hamiltonian that is to be diagonalized
    # accumulate basis transformations and eigenvalues: vectors, values
    # verbosity: text output in diagonalization
    ###############################
    if BlockOperator:
        basistrafos = [[],[]]
        basis_dict_set = []
    # initial bad guess for GS eneergy
    GS_temp_en = 10.
    # collect degenerate ground state qunatum numbers
    GS_QNset = []
    # tollerance to ground state energy fro GS set
    GS_tollerance = 1e-5
    # accumulating the quantum numbers
    Tot_QN_set = [qn for qn in hamil.H.qnind]
    # get the operators in a list
    operator_set = [[hamil.GetMat(QNind,False) for QNind,_ in enumerate(hamil.H.qnind)]]
    # here we write the intermediate
    tmp_operator_set = [[]]
    # only the first operator is explicitly needed
    for oind, operator in enumerate(SymmetrySet):
        operator_set += [[operator.GetMat(QNind,False) for QNind,_ in enumerate(operator.H.qnind)]]
        tmp_operator_set += [[]]
    for oind,_ in enumerate(operator_set):
        if oind < 1: continue
        # intermeidate operator set
        tmp_operator_set = [[]]
        for _, _ in enumerate(SymmetrySet):
            tmp_operator_set += [[]]
        # intermediate ground state set
        new_GS_QNset = []
        # eigenbasis
        val = []
        vec = []
        if len(SuperTrafo) == 0:
            for matrix in operator_set[oind]:
                tmp_val,tmp_vec = np.linalg.eigh(matrix)
                val += [tmp_val]
                vec += [tmp_vec]
        else:
            val = SuperTrafo[1][oind-1]
            vec = SuperTrafo[0][oind-1]
        # loop over parity eigenstates and energies
        for QNind,(QN,Pval,Pvec) in enumerate(zip(Tot_QN_set,val,vec)):
            #Pval,Pvec = np.linalg.eigh(test_par[j])
            Pval = np.round(Pval,2)
            # eliminate repeated quantum numbers
            uPval = np.unique(Pval)
            # map the omitted operators
            res_op_set = []
            res_op_dict = {}
            res_count = 0
            # bring reamining operators and Hamiltinian in Projector eigenbasis
            for res_oind, res_operator in enumerate(operator_set):
                if 0 < res_oind <= oind:
                    continue
                res_op_set += [np.dot(Pvec.T,np.dot(res_operator[QNind],Pvec))]
                # translating the temporary matrix ordering to the global ordering
                res_op_dict[res_count] = res_oind
                res_count += 1
            for i in range(uPval.size):
                #add key
                parent = QN+"S%.2f"%(uPval[i])
                # add parent QN to dict
                new_GS_QNset += [parent]
                # getting all indeces of the degenerate QN
                sub_index = np.where(Pval == uPval[i])[0]
                for mat_ind, matrix in enumerate(res_op_set):
                    # brign the Hamiltonian in the subspace
                    sub_op = (matrix[sub_index,:])[:,sub_index]
                    tmp_operator_set[res_op_dict[mat_ind]] += [sub_op]
        if BlockOperator:
            basistrafos[0] += [vec]
            basistrafos[1] += [val]
            tmp_basis_dict = {}
            for qnindex, QN in enumerate(Tot_QN_set):
                tmp_basis_dict[QN] = qnindex
            basis_dict_set += [tmp_basis_dict]
        operator_set = tmp_operator_set.copy()
        Tot_QN_set = new_GS_QNset.copy()
    ########
    # finally diagonalize the Hamiltonian
    ########
    # collect the hamiltonian eigenstates and energies
    par_ham_vec = []
    par_ham_val = []
    # diagonalize the Hamiltonian in the new basis
    # new QN dictionary
    new_ham_qn_dict = {}
    for index,(sub_ham,QN) in enumerate(zip(operator_set[0],Tot_QN_set)):
        ham_val, ham_vec = np.linalg.eigh(sub_ham)
        par_ham_vec += [ham_vec]
        par_ham_val += [ham_val]
        if verbosity: print(QN,"\n",ham_val,"\n")
        # creating a set containing the ground state quantum numbers
        if ham_val[0] < GS_temp_en-GS_tollerance:
            GS_temp_en = ham_val[0]
            GS_QNset = [QN]
        elif ham_val[0] < GS_temp_en+GS_tollerance and  ham_val[0] > GS_temp_en-GS_tollerance:
            GS_QNset += [QN]
        # add the new QN to the dicitonary
        new_ham_qn_dict[QN] = index
    # transforming back in to the original basis
    #print(np.dot(ham_vec.T,sub_trafo.T))
    print("GS quantum numbers: ",GS_QNset)
    if BlockOperator:
        return par_ham_val,par_ham_vec,new_ham_qn_dict,Tot_QN_set,basistrafos,basis_dict_set,GS_QNset
    return par_ham_val,par_ham_vec,new_ham_qn_dict,GS_QNset

def SuperBlockOperator(SuperTrafo,basis_dict_set,operator):
    ###############################
    # bringing an arbitrary operator in the eigenbasis of a block diagonalized Hamitlonian
    # SuperTrafo: quantum number basis generated from SuperBlockDiagonalization
    # basis_dict_set: dictionary connecting quantum numbers to block numbers
    # operator: operator object that is to be decomposed
    ###############################
    # collect creation anihilatiion operator blocks
    operator_set = [operator.GetMat(QNind,False) for QNind,_ in enumerate(operator.H.qnind)]
    # record the change in quantum numbers for every creation anihilatiion operator blocks
    qn_out = []
    # new dictionary for parity quantum number change
    new_qn_dict = {}
    for index,QN in enumerate(basis_dict_set[0]):
        QNint = [float(x) for x in QN[1::].split("S")]
        new_qn = "Q%.iS%.i"%(QNint[0]+operator.deltaQN[0],QNint[1]+operator.deltaQN[1])
        qn_out += [(QNint[0]+operator.deltaQN[0],QNint[1]+operator.deltaQN[1])]
        if new_qn in basis_dict_set[0]:
            new_qn_dict[QN] = [index]
    # loop over different basis trafos:
    for counting,(qnind,val,vec) in enumerate(zip(basis_dict_set,SuperTrafo[1],SuperTrafo[0])):
        # intermediate transformation
        intermeidate_operator_set = []
        # record the change in quantum numbers for every creation anihilatiion operator blocks
        tmp_qn_out = []
        # new dictionary for parity quantum number change
        tmp_new_qn_dict = {}
        qn_count = 0
        # loop over pairty eigen system
        for j,(QN,Pval,Pvec) in enumerate(zip(qnind,val,vec)):
            QNint = [float(x) for x in QN[1::].split("S")]
            # incoming QNind
            if not QN in new_qn_dict: continue
            QNind_in_set = new_qn_dict[QN]
            Pval_in = np.round(Pval,2)
            for qnout_ind_in in QNind_in_set:
                QN_out = "Q%.iS%.i"%(qn_out[qnout_ind_in][0],qn_out[qnout_ind_in][1])
                for addon in qn_out[qnout_ind_in][2::]:
                    QN_out += "S%.2f"%(addon)
                if not QN_out in qnind:
                    continue
                QNind_out = qnind[QN_out]
                Pval_out = np.round(val[QNind_out],2)
                # eliminate repeated quantum numbers
                uPval_in = np.unique(Pval_in)
                uPval_out = np.unique(Pval_out)
                # in
                Pvec_in = Pvec.T
                # out
                Pvec_out = np.transpose(vec[QNind_out].copy())
                # loop over qn combinations
                for i in range(uPval_in.size):
                    for k in range(uPval_out.size):
                        #add key corresponding to input quantum nuber
                        parent = QN+"S%.2f"%(uPval_in[i]) # takes ous to list that contains the full change of quantum nubers
                        # record the output quantum number
                        tmp_qn_out += [[*qn_out[qnout_ind_in],np.round(uPval_out[k],2)]]
                        if not parent in tmp_new_qn_dict:
                            tmp_new_qn_dict[parent] = [qn_count]
                            qn_count += 1
                        else:
                            tmp_new_qn_dict[parent] = [*tmp_new_qn_dict[parent].copy(),qn_count]
                            qn_count += 1
                        # getting all indeces of the degenerate QN
                        sub_index_in = np.where(Pval_in == uPval_in[i])[0]
                        #sub_trafo_in = Pvec_in[:,sub_index_in]
                        sub_trafo_in = Pvec_in[sub_index_in,:]
                        sub_index_out = np.where(Pval_out == uPval_out[k])[0]
                        #sub_trafo_out = Pvec_out[:,sub_index_out]
                        sub_trafo_out = Pvec_out[sub_index_out,:]
                        # bring operator in suspace
                        sub_operator = np.dot(sub_trafo_out,np.dot(operator_set[qnout_ind_in],sub_trafo_in.T))
                        # save the operator in a list
                        intermeidate_operator_set += [sub_operator]
        new_qn_dict = tmp_new_qn_dict
        qn_out = tmp_qn_out.copy()
        operator_set = intermeidate_operator_set.copy()
    return operator_set,new_qn_dict,qn_out
##########################################

##########################################    
# Initialize Hamiltonian class
class InitHam(object):
    def __init__(self, DOF, **kwargs):      
        # The amount of degrees of freedom are equal to
        # the number of sites.
        self.Nfermi = len(DOF)
        # Maximum charge Quanum number
        max_q = self.Nfermi
        # Maximum mgnetization Quanum number
        max_sz = self.Nfermi
        # making them globl
        self.max_q = max_q
        self.max_sz = max_sz
        
        # creating bonds
        if not "bonds" in kwargs:
            self.bnd = np.empty((0, 2), dtype='int')  # contains the bonds
            for si in range(self.Nfermi-1): #change range to N, for periodic boundary conditions
                self.bnd = np.vstack((self.bnd, np.array([si, si+1])))
            self.nb = self.bnd.shape[0]
        else:
            self.bnd = kwargs["bonds"]
            self.nb = len(self.bnd)
        # all sites
        
        # degrees of freeedom, e.g. [2,4] for a spin and an orbital
        if "boson_DOF" in kwargs:
            # number of bosons
            self.Nbosons = len(kwargs["boson_DOF"])
            self.N = self.Nfermi + self.Nbosons
            # here we create the full desnity of states
            self.DOF = DOF.copy() + kwargs["boson_DOF"]
            # the commutation relations we pretend that the bosons are spins
            # thus, we replace all boson DOFs with spin DOFs
            self.reducedDOF = DOF.copy() + [2 for i in kwargs["boson_DOF"]]
            # saving the bosons DOF
            self.bDOF =  kwargs["boson_DOF"]
        else:
            # no bosons are included
            self.DOF = DOF.copy() 
            self.reducedDOF = DOF.copy()
            self.Nbosons = 0
            # total number of particles
            self.N = len(DOF)
        ####################################################
        # setting the masis for state2num
        global num_bas
        num_bas = max(self.DOF)

        self.sites = np.arange(self.Nfermi)
        ####################################################
        # Creat the Hilbert space
        ###################################################
        # create a set containing all degrees of freedom  
        self.dof_set=[]
        for i in range(self.Nfermi):
            if DOF[i]==2: 
                self.dof_set += [[-1,1]] # Local moment
            else:
                self.dof_set += [[-1,0,1,2]] # Anderson type site
        for i in range(self.Nbosons):
            self.dof_set += [[j for j in range(self.bDOF[i])]] # Local moment
        
        # implementing restrictions on the Hilbertspace
        if "boundary_condition" in kwargs:
            # setting boundary conditions
            print("Using a reduced set of states")
            self.bc = kwargs["boundary_condition"]

            # create a set containing all possible states 
            self.full_basis = product(*self.dof_set,bcset=self.bc,repeat=1)
        else:
            # create a set containing all possible states 
            self.full_basis = it.product(*self.dof_set,repeat=1)
            self.bc = []

        # (qnind,r,site), where r denotes the index inside the QN space and qnind
        # is the quantum numbers subspace
        #basis = np.zeros((2*max_q+1,2*max_sz+1,self.msize,self.N+2),dtype=int)
        self.basis = [None for i in range((2*max_q+1)*(2*max_sz+1))]
        ###################################################
                 
        ###################################################
        # Create quantum number subspaces
        ###################################################
        # index maximum for each subspace  
        self.rmax = np.zeros((2*max_q+1)*(2*max_sz+1),dtype=int)

        # Quantum number dictionary returning the index associated
        # to the quantum number
        self.qnind = {}
        # state dictionary for indexing states
        self.statedic = {}

        # effective maximum quantum numbers (different from lower case variables)
        self.MaxSz = 0.
        self.MaxQ = 0.

        # checkpoint variables
        exists = False
        self.qnCount = 0
        # iterate over all quantum numbers
        # and states
        for state in self.full_basis:
            abs_state = np.abs(state[:self.Nfermi])
            q = np.sum(abs_state)-self.Nfermi  # calculate total charge; whith half filling = 0
            sz = np.sum((2-abs_state)*state[:self.Nfermi])
            # Check if QN is larger
            if sz > self.MaxSz: self.MaxSz = sz
            if q > self.MaxQ: self.MaxQ = q
            # updating index variable
            if not ("Q%.iS%.i"%(q,sz) in self.qnind):
                # Adding the index of the quantum number
                self.qnind["Q%.iS%.i"%(q,sz)] = self.qnCount
                # initializign qn state dict
                self.statedic[self.qnCount] = {}
                # initializing the stacking of the matrix
                stateArr = np.zeros((1,self.N+2),dtype=int)
                stateArr[0,1:self.N+1] = state[:]
                # initializing the basis
                self.basis[self.qnCount] = stateArr
                # getting the global index
                index = state2num(state)
                # adding the index to the dictionary
                self.statedic[self.qnCount][index] = self.rmax[self.qnCount]     
                # qunatum number index increment
                qnind = self.qnCount
                self.qnCount += 1            
            else:
                # get qn index
                qnind = self.qnind["Q%.iS%.i"%(q,sz)]
                # adding the state
                stateArr = np.zeros(self.N+2,dtype=int)
                stateArr[1:self.N+1] = state[:]
                # adding the state to the existing list
                self.basis[qnind] = np.vstack((self.basis[qnind],stateArr))
                # getting the global index
                index = state2num(state)
                # adding the index to the dictionary
                self.statedic[qnind][index] = self.rmax[qnind]   
            # incrementing the basis size
            self.rmax[qnind] += 1      
            # end for state
        ########## end initialization #############
        print("The Hilbert space of dimension ",np.sum(self.rmax)," is initialized")
    
    def State2Index(self,QNs,state):
        qnind = self.qnind[QNs]
        return self.statedic[qnind][state2num(state)]

                 
    def c(self, i, flavor, stateA):
        # Anihilation opeartor
        # - flavor: -1 = down, 1 = up spin 
        # - filling convention: 
        #     We fill down first -> down operators get always an extra sign when up is already filled
        #     The site filling goes from high to low
        # - the retunred `found` variable carries the commutation sign
        # - bosons should not contribute to `sign_count`
        
        if stateA[i] == flavor:
            stateB = np.copy(stateA)
            stateB[i] = 0
            sign_count = 0
            for j in range(0,i): # i is ommited and accounted for by `flavor`
                sign_count += abs(stateA[j])*(self.reducedDOF[j]//2 - 1) #calculate sign and account for bosons
            return (-1)**sign_count, stateB
        elif stateA[i] == 2:
            stateB = np.copy(stateA)
            stateB[i] = -1 * flavor
            sign_count = 0
            for j in range(0,i):
                sign_count += abs(stateA[j])*(self.DOF[j]//2 - 1)  
            return  flavor * (-1)**sign_count, stateB # the `flavor` sign accounts for the extra operator
        else:
            return 0, stateA

    def cd(self, i, flavor, stateA):
        # opeartor
        # - flavor: -1 = down, 1 = up spin 
        # - filling convention: 
        #     We fill down first -> down operators get always an extra sign when up is already filled
        #     The site filling goes from low to how
        # - the retunred `found` variable carries the commutation sign
        # - bosons should not contribute to `sign_count`
        
        if stateA[i] == 0:
            stateB = np.copy(stateA)
            stateB[i] = flavor
            sign_count = 0
            for j in range(0,i): # i is ommited and accounted for by `flavor`
                sign_count += abs(stateA[j])*(self.reducedDOF[j]//2 - 1) #calculate sign and account for bosons
            return (-1)**sign_count, stateB
        if stateA[i] == -1*flavor:
            stateB = np.copy(stateA)
            stateB[i] = 2
            sign_count = 0
            for j in range(0,i):
                sign_count += abs(stateA[j])*(self.reducedDOF[j]//2 - 1)
            return flavor * (-1)**sign_count, stateB
        else:
            return 0, stateA

    def cdc(self, flavor, i, j, stateA):
        # hopping operator: anihilation operator followed by creation operator
        # - returns sign and new state
        # - first we check that no boson is hopping

        if self.reducedDOF[j]*self.reducedDOF[j]<15:
            return 0, stateA

        found, stateB = self.c(j, flavor, stateA)
        sign = found
        if abs(found) == 1: 
            found, stateB = self.cd(i, flavor, stateB)
            sign *= found

        if abs(found) == 1:
            return sign, stateB
        else:
            return 0, stateA
        
    def b(self, i, stateA):
        # Bosnoic anihilation opeartor
        if stateA[i] > 0:
            stateB = np.copy(stateA)
            stateB[i] -= 1
            return np.sqrt(stateA[i]), stateB
        else:
            return 0, stateA
        
    def bd(self, i, stateA):
        # Bosnoic creation opeartor
        # The truncation introduces errors
        # we mitigated by setting the largest phonon energy to 0
        if stateA[i] < self.DOF[i]-1:
            stateB = np.copy(stateA)
            stateB[i] += 1
            return np.sqrt(stateB[i]), stateB
        else:
            return np.sqrt(stateA[i]+1), stateA
            #return 0., stateA
        
    def cdc_b(self, flavor, i, j, k, stateA):
        # hopping operator: anihilation operator followed by creation operator
        # - returns sign and new state
        # - first we check that no boson is hopping

        if self.reducedDOF[j]*self.reducedDOF[j]<15:
            return 0, stateA

        found, stateB = self.b(k, stateA)
        sign = found
        if abs(found) > 0.: 
            found, stateB = self.cdc(flavor, i, j, stateB)
            sign *= found

        if abs(found) > 0.:
            return sign, stateB
        else:
            return 0, stateA
        
    def cdc_bd(self, flavor, i, j, k, stateA):
        # hopping operator: anihilation operator followed by creation operator
        # - returns sign and new state
        # - first we check that no boson is hopping

        if self.reducedDOF[j]*self.reducedDOF[j]<15:
            return 0, stateA

        found, stateB = self.bd(k, stateA)
        sign = found
        if abs(found) > 0.: 
            found, stateB = self.cdc(flavor, i, j, stateB)
            sign *= found

        if abs(found) > 0:
            return sign, stateB
        else:
            return 0, stateA

    def Sp(self, i, stateA):
        # Spin rasing operator
        if stateA[i] == -1:
            stateB = np.copy(stateA)
            stateB[i] = 1
            return 1, stateB
        else:
            return 0, stateA

    def Sm(self, i, stateA):
        # Spin lowering operator
        if stateA[i] == 1:
            stateB = np.copy(stateA)
            stateB[i] = -1
            return 1, stateB
        else:
            return 0, stateA

    def Sz(self, i, stateA):
        # Magnetic spin component
        if abs(stateA[i]) == 1:
            return stateA[i] * 0.5
        else:
            return 0

    def SpSm(self, i, j, stateA):
        # Spin-flip operator:
        # - Site i: spin gets raised
        # - Site j: spin gets lowered
        found, stateB = self.Sm(j, stateA)
        if found == 1: found, stateB = self.Sp(i, stateB)

        if found == 1:
            return 1, stateB
        else:
            return 0, stateA

    def SmSp(self, i, j, stateA):
        # Spin-flip operator:
        # - Site j: spin gets raised
        # - Site i: spin gets lowered
        found, stateB = self.Sp(j, stateA)
        if found == 1: found, stateB = self.Sm(i, stateB)

        if found == 1:
            return 1, stateB
        else:
            return 0, stateA

    def getH(self, parameters, **kwargs): 
        # Input:  
        #    L1: is the number of bonds, 
        #    L2: sets the interaction range
        #    Qztot: total particle number
        #    Sztot: total magnetic quantum number
        #
        # First we collect all the states in the right subspace.
        # The Hamiltonian is a list of sub Hamiltonians in the QN subspaces
        # the QN subspace Hamiltonians have the `self.qnind` index
        
        ########################################
        # reading the fermion/spin parameters
        ########################################
        # possible Hamiltonian terms: ('name',shape)
        self.poptions = [("hubbard-U",1),("extended-hubbard-U",1),("epsilon",1),("hopping",1),("magfield",1),("spin-spin",2)]
        # parameters: dictionary containing the numpy arrays defining the interactions
        self.terms = {}
        for int_name in self.poptions:
            name, dim = int_name
            if name in parameters:
                self.terms[name] = parameters[name]
            else:
                if dim == 1: self.terms[name] = np.zeros(self.Nfermi)
                else: self.terms[name] = np.zeros((self.Nfermi,dim))
        
        # reading the boson parameters
        if self.Nbosons > 0:
            if not "boson_site_coupling" in kwargs:
                self.boson_sc = []
                print("Warning: no boson site coupling was specified")
            else:
                self.boson_sc = kwargs["boson_site_coupling"]
            if not "boson_bond_coupling" in kwargs:
                self.boson_bc = []
                print("Warning: no boson bond coupling was specified")
            else:
                self.boson_bc = kwargs["boson_bond_coupling"]
            if not "boson_omega" in kwargs:
                self.boson_om = [0. for i in self.bDOF]
                print("Warning: no boson bond coupling was specified")
            else:
                self.boson_om = kwargs["boson_omega"]
        else:
            self.boson_sc = []
            self.boson_bc = []
            self.boson_om = []
        
        ########################################
        # constructing the Hamiltonian
        ########################################
        
        H = [None for i in range(self.qnCount)]

        # checkpoint variable
        addham_qsz = False
        
        # We go through all the interactions to create the Hamiltonian 
        for q in np.arange(-self.MaxQ,self.MaxQ+1):
            for sz in np.arange(-self.MaxSz,self.MaxSz+1):

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.qnind): continue
                
                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number
                qnind = self.qnind["Q%.iS%.i"%(q,sz)] 
                
                # subspace Hamiltonian
                H_qsz = np.zeros((self.rmax[qnind],self.rmax[qnind]))
                for r in range(self.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.basis[qnind][r,1:self.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.basis[qnind][r,1:self.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    
                    # loop over boson bond coupling
                    for bbc in self.boson_bc:
                        si, sj = self.bnd[bbc[0]] # first index specifies the bond it couples to
                        bindex = self.Nfermi + bbc[1] # second index specifies which boson couples
                        b_lam = bbc[2] # coupling strength
                        # loop over spin
                        for sigma in [-1,1]:
                            # compute the matrix elements
                            found, stateB = self.cdc_bd(sigma, si, sj, bindex, BosonStateA)
                            if abs(found) > 0:# and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(stateB)]
                                H_qsz[r, j] += b_lam * found
                            found, stateB = self.cdc_b(sigma, si, sj, bindex, BosonStateA)
                            if abs(found) > 0:# and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(stateB)]
                                H_qsz[r, j] +=  b_lam * found
                            found, stateB = self.cdc_bd(sigma, sj, si, bindex, BosonStateA)
                            if abs(found) > 0:# and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(stateB)]
                                H_qsz[r, j] +=  b_lam * found
                            found, stateB = self.cdc_b(sigma, sj, si, bindex, BosonStateA)
                            if abs(found) > 0:# and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(stateB)]
                                H_qsz[r, j] += b_lam * found
                    # loop over boson site couplig
                    for bsc in self.boson_sc:
                        si = bsc[0] # first index specifies the bond it couples to
                        bindex = self.Nfermi + bsc[1] # second index specifies which boson couples
                        b_lam = bsc[2] # coupling strength
                        # loop over spin
                        for sigma in [-1,1]:
                            # compute the matrix elements
                            found, stateB = self.cdc_bd(sigma, si, si, bindex, BosonStateA)
                            if abs(found) > 0.:# and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(stateB)]
                                H_qsz[r, j] += b_lam * found
                            found, stateB = self.cdc_b(sigma, si, si, bindex, BosonStateA)
                            if abs(found) > 0.:# and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(stateB)]
                                H_qsz[r, j] += b_lam * found
                    
                    # diagonal value
                    diag_value = 0.
                    # loop over bonds for fermionc / spin interactions
                    for b in range(self.nb):
                        si, sj = self.bnd[b]
                        bondJz, bondJortho = self.terms["spin-spin"][b] # spin coupling of bond `b`
                        bond_t = self.terms["hopping"][b] # hopping of bond `b`

                        # Spin-spin interaction Hamiltonian
                        # calculate the off-diagonal Hamiltonian part
                        found, stateB = self.SpSm(si, sj, stateA)
                        if found == 1:# and state2num(stateB) in self.statedic[qnind]:
                            j = self.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.Nfermi::])))]
                            H_qsz[r, j] +=  bondJortho * 0.5
                        found, stateB = self.SmSp(si, sj, stateA)
                        if found == 1:# and state2num(stateB) in self.statedic[qnind]:
                            j = self.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.Nfermi::])))]
                            H_qsz[r, j] +=  bondJortho * 0.5
                       
                        # extended Hubbard interaction
                        exU = self.terms['extended-hubbard-U'][b]
                        diag_value += exU*abs_state[si]*abs_state[sj]

                        #calculate the diagonal Hamiltonian part
                        #Note: duplicate indices are maintained until implicitly or explicitly summed
                         
                        diag_value +=  bondJz * self.Sz(si, stateA) * self.Sz(sj, stateA)

                        # Hopping term:
                        for spin in [-1,1]:
                            # we do the hopping
                            found, stateB = self.cdc(spin, si, sj, stateA)
                            if abs(found) == 1: # and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.Nfermi::])))]
                                H_qsz[r, j] +=  bond_t * found
                            found, stateB = self.cdc(spin, sj, si, stateA)
                            if abs(found) == 1: # and state2num(stateB) in self.statedic[qnind]:
                                j = self.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.Nfermi::])))]
                                H_qsz[r, j] += bond_t * found
                    
                    # Magnetic field
                    diag_value += 0.5*np.dot(self.terms["magfield"],(2.-abs_state)*stateA)
                    # Density interaction and chemical potential
                    # calculate the diagonal Hamiltonian part
                    diag_value += np.dot(abs_state,self.terms["epsilon"]) 
                    diag_value += 0.5*np.dot((abs_state - 1)*stateA,self.terms["hubbard-U"])
                    # boson energies
                    for bnum, freq in enumerate(self.boson_om):
                        diag_value += BosonStateA[self.Nfermi+bnum] * freq
                    # collect the diagonal
                    H_qsz[r, r] +=  diag_value
                # end for r
                # adding the QN subspace Hamiltonain if the QN combination exist
                if addham_qsz:
                    H[qnind] = sparse.csr_matrix(H_qsz)
                    addham_qsz = False
            # end for sz
        # end for q
        # Create Hamiltonian operator
        HamOperator = OperatorBase(self, H, [0,0]) 
        return HamOperator

class Operators(object):
    def __init__(self,H,**kwargs):
        ##############################
        # If the projectors are requested this function returns
        # randomly initialized angles and dummy observalbes list
        # adding the Hamiltonian to the internal structure
        self.H = H
        # full basis Hamiltonian for state reference
        try:
            self.refH = kwargs['refH']
            self.refHbool = True
        except:
            self.refHbool = False

    def InitPrj(self):
        return self.obsind, self.angles, self.gradobsind

    def SzIJ(self,flavor,i,j, stateA):
        # Magnetic spin component
        found, stateB = self.cdUcD(flavor,flavor,i,j,stateA)
        if abs(found) == 1:
            return found * 0.5, stateB
        else:
            return 0, stateA

    def SpSmIJ(self, i, j, k, stateA):
        # Spin-flip operator:
        # - Site i: spin gets raised
        # - Site j: spin gets lowered
        found, stateB = self.cdUcD(-1,1,i,j,stateA)
        sign = found
        if abs(found) == 1: found, stateB = self.H.Sp(k, stateB)

        if abs(found) == 1:
            return sign, stateB
        else:
            return 0, stateA

    def SmSpIJ(self, i, j, k, stateA):
        # Spin-flip operator:
        # - Site j: spin gets raised
        # - Site i: spin gets lowered
        found, stateB = self.cdUcD(1,-1,i,j,stateA)
        sign = found
        if abs(found) == 1: found, stateB = self.H.Sm(k, stateB)

        if abs(found) == 1:
            return sign, stateB
        else:
            return 0, stateA

    def SiSjk(self,central,bond):
        # Spin-spin exchange term between the bonds [i,j]
        # with corss hopping
        # bond: cross hopping spin
        # central: not cross hopping spin

        si = bond[0]
        sj = bond[1]
        
        # List to accumulate the subspaces
        Obs = [None for i in range(self.H.qnCount)]

        # checkpoint variable
        addham_qsz = False

        # We go through all the interactions to create the observable
        for q in np.arange(-self.H.max_q,self.H.max_q+1):
            for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):
                
                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue
                
                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                
                # subspace Hamiltonian
                Obs_qsz = np.zeros((self.H.rmax[qnind],self.H.rmax[qnind]))

                for r in range(self.H.rmax[qnind]):
                    # Destroying a spin down in one lead creates a spin in another lead
                    found, stateB = self.SmSpIJ(si, sj, central, self.H.basis[qnind][r,1:self.H.N+1])
                    if abs(found) == 1:
                        j = self.H.statedic[qnind][state2num(stateB)]
                        Obs_qsz[r,j] += found  * 0.5
                    found, stateB = self.SpSmIJ(si, sj, central, self.H.basis[qnind][r,1:self.H.N+1])
                    if abs(found) == 1:
                        j = self.H.statedic[qnind][state2num(stateB)]
                        Obs_qsz[r,j] += found  * 0.5
                    found, stateB = self.SzIJ(1,si, sj, self.H.basis[qnind][r,1:self.H.N+1])
                    if abs(found) == 0.5:
                        found *= self.H.Sz(central, stateB)
                        j = self.H.statedic[qnind][state2num(stateB)]
                        Obs_qsz[r,j] += found
                    found, stateB = self.SzIJ(-1,si, sj, self.H.basis[qnind][r,1:self.H.N+1])
                    if abs(found) == 0.5:
                        found *= self.H.Sz(central, stateB)
                        j = self.H.statedic[qnind][state2num(stateB)]
                        Obs_qsz[r,j] -= found
#                 Obs_qsz += Obs_qsz.T
#                 Obs_qsz -= np.diag(np.diag(Obs_qsz))/2.
                # end for r
                # adding the QN subspace Observable if the QN combination exists
                if addham_qsz:
                    Obs[qnind] = Obs_qsz
                    addham_qsz = False
            # end for sz
        # end for q     
        Operator = OperatorBase(self.H, Obs, [0,0]) 
        return Operator 

    def Stot(self):
        # Total spin opeartor for the impurity system

        H = [None for i in range(self.H.qnCount)]

        # defining the specific bonds
        bonds = list(it.combinations(range(self.H.Nfermi),r=2))

        for q in np.arange(-self.H.max_q,self.H.max_q+1):
            for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue

                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 

                # subspace Hamiltonian
                H_qsz = np.zeros((self.H.rmax[qnind],self.H.rmax[qnind]))

                for r in range(self.H.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    # tot-spin
                    si_tot = 0.
                    for bnd in bonds:
                        si, sj = bnd
                        found, stateB = self.H.SpSm(si, sj, stateA)
                        if found == 1:
                            j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                            H_qsz[r, j] +=  1.
                        found, stateB = self.H.SmSp(si, sj, stateA)
                        if found == 1:
                            j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                            H_qsz[r, j] +=  1.
                        si_tot += 2. * self.H.Sz(si, stateA) * self.H.Sz(sj, stateA)
                    si_tot += np.sum(np.mod(abs_state,2)) * 0.5*(1.+0.5)
                    H_qsz[r, r] += si_tot
                # end for r
                # adding the QN subspace Hamiltonain if the QN combination exists
                if addham_qsz:
                    H[qnind] = sparse.csr_matrix(H_qsz)
                    addham_qsz = False
            # end for sz
        # end for q     
        Operator = OperatorBase(self.H, H, [0,0]) 
        return Operator
    
    def Tij(self,bond): 
        # Hoopping terms:
        # takes the bonds [i,j] 
        # The hopping amplitude is `found` equal to 1

        si = bond[0]
        sj = bond[1]

        # List to accumulate the subspaces
        Obs = [None for i in range(self.H.qnCount)]

        # checkpoint variable
        addham_qsz = False

        # We go through all the interactions to create the observable
        for q in np.arange(-self.H.max_q,self.H.max_q+1):
            for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):
                
                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue
                
                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                
                # subspace Hamiltonian
                Obs_qsz = np.zeros((self.H.rmax[qnind],self.H.rmax[qnind]))

                for r in range(self.H.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    # tot-spin
                    # for up and down spin
                    for spin in [-1,1]:
                        # getting the bra state after applzying the interaction
                        found, stateB = self.H.cdc(spin, si, sj, stateA)
                        if abs(found) == 1:
                            j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                            Obs_qsz[r,j] += found     
                        found, stateB = self.H.cdc(spin, sj, si, stateA)
                        if abs(found) == 1:
                            j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                            Obs_qsz[r,j] += found
                # end for r
                # adding the QN subspace Observable if the QN combination exists
                if addham_qsz:
                    Obs[qnind] = sparse.csr_matrix(Obs_qsz)
                    addham_qsz = False
            # end for sz
        # end for q
        Operator = OperatorBase(self.H, Obs, [0,0]) 
        return Operator

    def cdUcD(self,flavori,flavorj,i,j,stateA):
        # general-spin flip operator: anihilation operator followed by creation operator
        # with different spin flavours
        # - returns sign and new state
        # - first we check that no boson is hopping

        if self.H.DOF[i]*self.H.DOF[j]<15:
            return 0, stateA

        found, stateB = self.H.c(j, flavorj, stateA)
        sign = found
        if abs(found) == 1: 
            found, stateB = self.H.cd(i, flavori, stateB)
            sign *= found

        if abs(found) == 1:
            return sign, stateB
        else:
            return 0, stateA

    def Sij(self,bond):
        # Spin-spin exchange term between the bonds [i,j]

        si = bond[0]
        sj = bond[1]
        
        # List to accumulate the subspaces
        Obs = [None for i in range(self.H.qnCount)]

        # checkpoint variable
        addham_qsz = False

        # We go through all the interactions to create the observable
        for q in np.arange(-self.H.max_q,self.H.max_q+1):
            for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):
                
                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue
                
                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                
                # subspace Hamiltonian
                Obs_qsz = np.zeros((self.H.rmax[qnind],self.H.rmax[qnind]))

                for r in range(self.H.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)                    
                    """ for up and down spin """
                    found, stateB = self.H.SpSm(si, sj, stateA)
                    if found == 1:
                        j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                        Obs_qsz[r,j] += found  * 0.5
                    found, stateB = self.H.SmSp(si, sj, stateA)
                    if found == 1:
                        j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                        Obs_qsz[r,j] += found  * 0.5

                    Obs_qsz[r,r] = self.H.Sz(si,stateA) * self.H.Sz(sj, stateA)
                # end for r
                # adding the QN subspace Observable if the QN combination exists
                if addham_qsz:
                    Obs[qnind] = sparse.csr_matrix(Obs_qsz)
                    addham_qsz = False
            # end for sz
        # end for q
        Operator = OperatorBase(self.H, Obs, [0,0]) 
        return Operator

    def creation(self,site,flavour): 
        # Input:  
        #    site: site to act with the opeator 
        #    flavor: flavour to act upon
        # Output:
        #     creation operator: list of rank 2 matrices
        #     F operator: list of rank 2 matrices
        #     list: this list specifies how this operator cahnges the Quantum numbers
        
        if not self.H.DOF[site] == 4 or site > self.H.Nfermi:
            raise("This operator is only defined for fermions")
            return None
         
        flavdict = {"up":1,"down":-1}
        max_q = self.H.max_q
        max_sz = self.H.max_sz
        
        if not flavour in flavdict:
            raise("please specify: flavour = up,down")
            return None

        # The Hamiltonian is a list of sub Hamiltonians in the QN subspaces
        # the QN subspace Hamiltonians have the `self.qnind` index
        cd = [0 for i in range(self.H.qnCount)]
        
        # We go through all the interactions to create the Hamiltonian 
        for q in np.arange(-max_q,max_q+1):
            for sz in np.arange(-max_sz,max_sz+1):

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue

                # getting the index of the ket QN
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q+1,sz+flavdict[flavour]) in self.H.qnind): 
                    cd[qnind] = sparse.csr_matrix(np.zeros((self.H.rmax[qnind],self.H.rmax[qnind])))
                    continue

                # getting the index of the bra QN
                qnind_new = self.H.qnind["Q%.iS%.i"%(q+1,sz+flavdict[flavour])] 
                
                # subspace Hamiltonian
                H_qsz = np.zeros((self.H.rmax[qnind_new],self.H.rmax[qnind]))

                for r in range(self.H.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    # creation operator
                    found, stateB = self.H.cd(site, flavdict[flavour], stateA)
                    if abs(found) == 1:
                        j = self.H.statedic[qnind_new][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                        H_qsz[j,r] += found
                # adding the QN subspace Hamiltonain if the QN combination exists
                cd[qnind] = sparse.csr_matrix(H_qsz)
            # end for sz
        # end for q
        Operator = OperatorBase(self.H, cd, [+1,flavdict[flavour]]) 
        return Operator

    def anihilation(self,site,flavour): 
        # Input:  
        #    site: site to act with the opeator 
        #    flavor: flavour to act upon
        # Output:
        #     anihilation operator: list of rank 2 matrices
        #     F operator: list of rank 2 matrices
        #     list: this list specifies how this operator cahnges the Quantum numbers
        
        if not self.H.DOF[site] == 4 or site > self.H.Nfermi:
            print("This operator is only defined for fermions")
            return None
         
        flavdict = {"up":1,"down":-1}
        max_q = self.H.max_q
        max_sz = self.H.max_sz
        
        if not flavour in flavdict:
            print("please specify: flavour = up,down")
            return None

        # The Hamiltonian is a list of sub Hamiltonians in the QN subspaces
        # the QN subspace Hamiltonians have the `self.qnind` index
        c = [0 for i in range(self.H.qnCount)]
        
        # We go through all the interactions to create the Hamiltonian 
        for q in np.arange(-max_q,max_q+1):
            for sz in np.arange(-max_sz,max_sz+1):

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue

                # getting the index of the ket QN
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q-1,sz-flavdict[flavour]) in self.H.qnind): 
                    c[qnind] = sparse.csr_matrix(np.zeros((self.H.rmax[qnind],self.H.rmax[qnind])))
                    continue

                # getting the index of the bra QN
                qnind_new = self.H.qnind["Q%.iS%.i"%(q-1,sz-flavdict[flavour])] 
                
                # subspace Hamiltonian
                H_qsz = np.zeros((self.H.rmax[qnind_new],self.H.rmax[qnind]))

                for r in range(self.H.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    # anihilation operator
                    found, stateB = self.H.c(site, flavdict[flavour], stateA)
                    if abs(found) == 1:
                        j = self.H.statedic[qnind_new][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                        H_qsz[j,r] += found
                c[qnind] = sparse.csr_matrix(H_qsz)
            # end for sz
        # end for q
        Operator = OperatorBase(self.H, c, [-1,-flavdict[flavour]]) 
        return Operator

    def UiEi(self,site,reps,rU):
        # Density interaction on the site `site` 
        # reps: chemical potential
        # rU: interaction strength
        
        U = np.zeros(self.H.N)
        eps = np.zeros(self.H.N)
        U[site] = rU
        eps[site] = reps
        
        # List to accumulate the subspaces
        Obs = [None for i in range(self.H.qnCount)]

        # checkpoint variable
        addham_qsz = False

        # We go through all the interactions to create the observable
        for q in np.arange(-self.H.max_q,self.H.max_q+1):
            for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):
                
                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue
                
                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                
                # subspace Hamiltonian
                Obs_qsz = np.zeros((self.H.rmax[qnind],self.H.rmax[qnind]))
                # looping over all states
                for r in range(self.H.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    # getting the matrix element
                    Obs_qsz[r,r] = np.dot(abs_state,eps) + 0.5*np.dot((abs_state - 1)*stateA,U)
                # end for r
                # adding the QN subspace Hamiltonain if the QN combination exists
                if addham_qsz:
                    Obs[qnind] = sparse.csr_matrix(Obs_qsz)
                    addham_qsz = False
            # end for sz
        # end for q
        Operator = OperatorBase(self.H, Obs, [0,0]) 
        return Operator
    
    def Sminus(self):
        # Spin lowering opeartor
        # Bonds: lsit of sites of sites to which
        # the operator applies
        # NOTE: this operator connects different spin subspcaes

        H = [0 for i in range(self.H.qnCount)]

        for q in np.arange(-self.H.max_q,self.H.max_q+1):
            for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):

                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue

                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number (ket)
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 

                sz_new = sz - 2
                # check if the new quantum number exists
                if not ("Q%.iS%.i"%(q,sz_new) in self.H.qnind): continue
                # getting the index of the new quantum number (bra)
                qnind_new = self.H.qnind["Q%.iS%.i"%(q,sz_new)] 
                

                # subspace Hamiltonian
                H_qsz = np.zeros((self.H.rmax[qnind],self.H.rmax[qnind_new]))

                for r in range(self.H.rmax[qnind]):                  
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    for si in range(self.H.N):
                        #si = bnd
                        found, stateB = self.H.Sm(si, stateA)
                        if found == 1:
                            j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                            H_qsz[r,j] += 1.
                # end for r
                # adding the QN subspace Hamiltonain if the QN combination exists
                if addham_qsz:
                    Obs[qnind] = sparse.csr_matrix(Obs_qsz)
                    addham_qsz = False
            # end for sz
        # end for q
        Operator = OperatorBase(self.H, Obs, [0,-1]) 
        return Operator
    
    def Parity(self,par_dict,print_index=False):
        # Partiy operator switches the sites indeces according to a symmetry axis
        # the symmetry axis is defined in par_dict
        # par_dict = {site_1:site_i_1,site_2:site_i_2,...}

        # test ordering
        for p in par_dict:
            p_switch = par_dict[p]
            if p_switch < p:
                print("Error: Parity operator is not defined correctly")
                return None
        
        # List to accumulate the subspaces
        Obs = [None for i in range(self.H.qnCount)]

        # checkpoint variable
        addham_qsz = False

        # We go through all the interactions to create the observable
        for q in np.arange(-self.H.max_q,self.H.max_q+1):
            for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):
                
                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue
                
                # setting checkpoint
                addham_qsz = True

                # getting the index of the quantum number
                qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                
                # subspace Hamiltonian
                Obs_qsz = np.zeros((self.H.rmax[qnind],self.H.rmax[qnind]))
                # looping over all states
                for r in range(self.H.rmax[qnind]):
                    # only spins and fermions
                    stateA = self.H.basis[qnind][r,1:self.H.Nfermi+1]
                    abs_state = np.abs(stateA)
                    # spins, fermions and bosons
                    BosonStateA = self.H.basis[qnind][r,1:self.H.N+1]
                    BosonAbs_State = np.abs(BosonStateA)
                    # ouput sign
                    out_sign = 1.
                    # temporary state
                    stateB = stateA.copy()
                    # switching the sites according to the parity dictionary
                    for p in par_dict:
                        p_switch = par_dict[p]
                        tmp = stateB[p].copy()
                        stateB[p] = stateB[p_switch].copy()
                        stateB[p_switch] = tmp.copy()
                        sign_state = np.array(np.sin(np.abs(stateB)*np.pi/2.),dtype=int)
                        sign_a = sign_state[p]*np.sum(sign_state[p+1:p_switch+1])
                        sign_b = sign_state[p_switch]*np.sum(sign_state[p+1:p_switch])
                        out_sign *= (-1.)**(sign_a+sign_b)
                    j = self.H.statedic[qnind][state2num(np.concatenate((stateB,BosonStateA[self.H.Nfermi::])))]
                    if print_index: print("index: ",r,j)
                    Obs_qsz[r,j] += out_sign
                # end for r
                # adding the QN subspace Hamiltonain if the QN combination exists
                if addham_qsz:
                    Obs[qnind] = sparse.csr_matrix(Obs_qsz)
                    addham_qsz = False
            # end for sz
        # end for q
        Operator = OperatorBase(self.H, Obs, [0,0]) 
        return Operator
    
##########################################    
class CalcCond(object):
    def __init__(self,SysObs,left_hyb ,right_hyb):
        #############
        # input:
        # left_hyb: [V1,V2,...V3]
        # right_hyb: [V1,V2,...V3]
        # SysObs: system observable generator
        #############
        # check hyb lists
        if not len(left_hyb) == SysObs.H.Nfermi:
            print("Hybrdization list illegal!")
            return None
        if not len(right_hyb) == SysObs.H.Nfermi:
            print("Hybrdization list illegal!")
            return None
        # using hybridization to create the "frontier" creation anihilation operators
        # left up
        self.left_cd_up = SysObs.creation(0,"up")
        self.left_cd_up.ScalarDot(left_hyb[0])
        self.left_c_up = SysObs.anihilation(0,"up")
        self.left_c_up.ScalarDot(left_hyb[0])
        # left down
        self.left_cd_do = SysObs.creation(0,"down")
        self.left_cd_do.ScalarDot(left_hyb[0])
        self.left_c_do = SysObs.anihilation(0,"down")
        self.left_c_do.ScalarDot(left_hyb[0])
        # right up
        self.right_cd_up = SysObs.creation(0,"up")
        self.right_cd_up.ScalarDot(right_hyb[0])
        self.right_c_up = SysObs.anihilation(0,"up")
        self.right_c_up.ScalarDot(right_hyb[0])
        # left down
        self.right_cd_do = SysObs.creation(0,"down")
        self.right_cd_do.ScalarDot(right_hyb[0])
        self.right_c_do = SysObs.anihilation(0,"down")
        self.right_c_do.ScalarDot(right_hyb[0])
        # accumulate frontier orbital
        for i,V in enumerate(left_hyb):
            if i == 0: continue
            # left up
            cd_up = SysObs.creation(i,"up")
            cd_up.ScalarDot(V)
            self.left_cd_up.OperatorSum(cd_up)
            c_up = SysObs.anihilation(i,"up")
            c_up.ScalarDot(V)
            self.left_c_up.OperatorSum(c_up)
            # left down
            cd_do = SysObs.creation(i,"down")
            cd_do.ScalarDot(V)
            self.left_cd_do.OperatorSum(cd_do)
            c_do = SysObs.anihilation(i,"down")
            c_do.ScalarDot(V)
            self.left_c_do.OperatorSum(c_do)
        for i,V in enumerate(right_hyb):
            if i == 0: continue
            # right up
            cd_up = SysObs.creation(i,"up")
            cd_up.ScalarDot(V)
            self.right_cd_up.OperatorSum(cd_up)
            c_up = SysObs.anihilation(i,"up")
            c_up.ScalarDot(V)
            self.right_c_up.OperatorSum(c_up)
            # right down
            cd_do = SysObs.creation(i,"down")
            cd_do.ScalarDot(V)
            self.right_cd_do.OperatorSum(cd_do)
            c_do = SysObs.anihilation(i,"down")
            c_do.ScalarDot(V)
            self.right_c_do.OperatorSum(c_do)
    
    def GammaMat(self,energies,states,gate_voltage,temp):
        # obtaining the fermi-dirac distribution
        fd = fermi_parent(temp)
        # create full partition function
        Z0 = sum([np.sum(np.exp(-x[1]/temp)) for x in energies])
        # equilibirum boltzmann distribution
        init_prob = [np.exp(-x[1]/temp)/Z0 for x in energies]
        # equilibrium dot-distribution
        P_init = np.array([])
        for vec in init_prob:
            P_init = np.concatenate((P_init,vec))
        # shape set
        shape_set = [x.shape[0] for x in init_prob]
        Gamma_Mat = np.zeros(2*P_init.shape)
        Gamma_in = np.zeros(2*P_init.shape)
        Gamma_out = np.zeros(2*P_init.shape)
        # get transition rate
        for i,ket in enumerate(states):
            for j,bra in enumerate(states):
                # getting the index range
                if i > 0:
                    dx = sum(shape_set[:i])
                else:
                    dx = 0
                if j > 0:
                    dy = sum(shape_set[:j])
                else:
                    dy = 0
                # initialize y
                y_in = 0.
                p_in = 0.
                y_out = 0.
                p_out = 0.
                # bath distribution
                bath_distro = 2. *np.pi * fd(energies[j][1] - energies[i][1].reshape(-1,1)-gate_voltage)
                drain_distro = 2. *np.pi  * fd(energies[j][1] - energies[i][1].reshape(-1,1)+gate_voltage)
                # calculate transition matrix
                left_out_up = self.left_cd_up.BraKetDot(bra,ket)
                if not left_out_up[0] == None:
                    right_out_up = self.right_cd_up.BraKetDot(bra,ket)
                    # compute apmplitude
                    y_out += bath_distro.T*(np.abs(left_out_up[1])**2)
                    p_out += drain_distro.T*(np.abs(right_out_up[1])**2)
                # calculate transition matrix: 
                left_out_do = self.left_cd_do.BraKetDot(bra,ket)
                if not left_out_do[0] == None:
                    right_out_do = self.right_cd_do.BraKetDot(bra,ket)
                    # compute apmplitude
                    y_out += bath_distro.T*(np.abs(left_out_do[1])**2)
                    p_out += drain_distro.T*(np.abs(right_out_do[1])**2)
                # gamma in 
                Gamma_out[dy:dy+shape_set[j],dx:dx+shape_set[i]] += y_out
                # bath distribution
                bath_distro = 2. * np.pi * (1. - fd(energies[i][1] - energies[j][1].reshape(-1,1)-gate_voltage))
                drain_distro = 2. * np.pi * (1. - fd(energies[i][1] - energies[j][1].reshape(-1,1)+gate_voltage))
                # calculate transition matrix
                left_out_up = self.left_c_up.BraKetDot(bra,ket)
                if not left_out_up[0] == None:
                    right_out_up = self.right_c_up.BraKetDot(bra,ket)
                    # compute apmplitude
                    y_in += bath_distro*(np.abs(left_out_up[1])**2)
                    p_in += drain_distro*(np.abs(right_out_up[1])**2)
                # calculate transition matrix
                left_out_do = self.left_c_do.BraKetDot(bra,ket)
                if not left_out_do[0] == None:
                    right_out_do = self.right_c_do.BraKetDot(bra,ket)
                    # compute apmplitude
                    y_in += bath_distro*(np.abs(left_out_do[1])**2)
                    p_in += drain_distro*(np.abs(right_out_do[1])**2)
                # gamma out
                Gamma_in[dy:dy+shape_set[j],dx:dx+shape_set[i]] = y_in
                # all gamma
                Gamma_Mat[dy:dy+shape_set[j],dx:dx+shape_set[i]] = y_in+y_out+p_in+p_out
        Gamma_Mat = Gamma_Mat-np.diag(np.sum(Gamma_Mat,axis=0))
        return Gamma_Mat,Gamma_in,Gamma_out,P_init
        
    def rk4(self, Lin, rho0, N=100):
        # This function implements a runge kutta of 4th order
        # ===================================================
        # input:
        # Lin: we take the linblad operator, which is a numpy matrix.
        # tspan: touple, which contains tart and endpoint. 
        # rho0: initial density matrix of numpy vecotr type
        # ===================================================
        # output:
        # time_sequence: numpy array of time steps
        # rho_sequence: python list of numpy vecotrs containing 
        # the density matrix
        # ===================================================
        start = timeit.default_timer()
        # initial stepsize
        h = 1e-5
        rho = rho0
        print(f'{"> start":->10}')
        for i in range(0,N):
            k1 = Lin.dot(rho)
            k2 = Lin.dot(rho + 0.5 * h*k1)
            k3 = Lin.dot(rho + 0.5 * h*k2)
            k4 = Lin.dot(rho + h*k3)
            rho = rho + h/6. * (k1 + 2. * (k2 + k3) + k4)
            rho /= np.sum(rho)
            tau = sum(abs(Lin.dot(rho)))
            # adaptive step size
            h = min([(1e-3*(1./tau)**2)**0.5,1.])
            if sum(abs(Lin.dot(rho))) < 1e-11: 
                break
        print(f'{"> done":->25}')
        stop = timeit.default_timer()
        m, s = divmod(stop-start, 60)
        h, m = divmod(m, 60)
        print('Runtime: %.0d:%.2d:%.2d'%(h, m, s))
        return rho, sum(abs(Lin.dot(rho)))
    
    def MatExp(self,Lin,FTime,rho):
        # Matrix exponential to
        # solve linear ODE 
        trafo = sp.linalg.expm(FTime*Lin)
        P_final = np.dot(trafo,rho)
        return rho/sum(rho), sum(abs(Lin.dot(rho)))
    
    def Current(self,Gamma_in,Gamma_out,P_final):
        # P_final: non-equilibrium distribtuion
        # Gamma_Out: matrix
        # Gamma_In: matrix
        return -np.sum(np.dot(Gamma_out - Gamma_in,P_final))

###########################################
class Sys2HDF5(object):
    def __init__(self,H,ham,hyb,**kwargs):
        # defiing default directory to run the NRG
        try:
            self.dir = kwargs['rundir']
        except:
            raise Exception("Specifiy a run directory for the NRG: ['rundir'] = <my_directory>")      
        # checking if directory exists
        if not (os.path.exists(self.dir)):
            # make directory
            os.mkdir(self.dir)       
        if not self.dir[-1] == "/":
            self.dir = self.dir+"/"    
        # adding the Hamiltonian object to self
        self.H = H
        # filename
        fname = self.dir+"InitSystem.hdf5"
        # open the file
        self.f = h5py.File(fname, "w")
        # enforcing symmetry attribute
        self.f.attrs['szsym'] = kwargs['szsym']
        self.f.attrs['qsym'] = kwargs['qsym']
        # wrting the amount of sites in the system
        self.f.attrs['sysites'] = H.N
        # writing the the starting point of the wilson chain
        self.f.attrs['lmin'] = kwargs['lmin']
        # getting the number of QN subspaces
        self.f.attrs['QNs'] = np.count_nonzero(H.rmax!=0)
        
        # ------ rmax ------
        rmax_grp = self.f.create_group("rmax")
        rmax_grp.attrs['MaxQ'] = H.MaxQ
        rmax_grp.attrs['MaxSz'] = H.MaxSz
        for q in np.arange(-H.MaxQ,H.MaxQ+1):
            for sz in np.arange(-H.MaxSz,H.MaxSz+1):
                # Skip QN combinations that do not contain any states
                if "Q%.iS%.i"%(q,sz) in H.qnind:
                    qnind = H.qnind["Q%.iS%.i"%(q,sz)]
                    rmax_grp.attrs["Q%.iS%.i"%(q,sz)] = H.rmax[qnind]
                else:
                    rmax_grp.attrs["Q%.iS%.i"%(q,sz)] = 0
        
        # ------ Hamiltonian ------
        # creating hamiltonian group
        ham_grp = self.f.create_group("Hamiltonian")
        # saving the qn subspaces to hdf5
        for q in np.arange(-H.max_q,H.max_q+1):
            for sz in np.arange(-H.max_sz,H.max_sz+1):
                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in H.qnind): continue
                # getting the index of the quantum number
                qnind = H.qnind["Q%.iS%.i"%(q,sz)] 
                # writing the dataset
                dset = ham_grp.create_dataset("Q%.iS%.i"%(q,sz), data = np.array(ham[qnind]).T,dtype=np.double)

        # ------ Basis ------
        # creating the basis group
        basis_grp = self.f.create_group("Basis")
        basis_grp.attrs['BasisDim'] = H.maxind
        # the basis must be transposed to be conformative with the Fortran 
        # column - row dominance convention
        # further we need to shift the site index down by one to fit the Fortran convention
        for q in np.arange(-H.max_q,H.max_q+1):
            for sz in np.arange(-H.max_sz,H.max_sz+1):
                # Skip QN combinations that do not contain any states
                if not ("Q%.iS%.i"%(q,sz) in H.qnind): continue
                # get index
                qnind = H.qnind["Q%.iS%.i"%(q,sz)]
                # writing the dataset
                savebasis = np.zeros((H.maxind,H.maxind))
                savebasis[:H.rmax[qnind],:H.N+2] = np.roll(H.basis[qnind],-1,axis=1)
                dset = basis_grp.create_dataset("Q%.iS%.i"%(q,sz), data=savebasis.T,dtype='int')
    
    def Observables(self,ObsList):
        # Saving the obserables
        # intput:
        #    list of list of 2D arrays - type double
        # ------ Observables ------
        obs_grp = self.f.create_group("Observables")
        # amount of observables
        NObs = len(ObsList)
        self.NObs = NObs
        # ---> storing to attribute
        obs_grp.attrs['NObs'] = NObs
        for i in range(NObs):
            oi_group = obs_grp.create_group("Obs%.i"%(i))
            Obs = ObsList[i]
            for q in np.arange(-self.H.max_q,self.H.max_q+1):
                for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):
                    # Skip QN combinations that do not contain any states
                    if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue
                    # getting the index of the quantum number
                    qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                    # writing the dataset
                    dset = oi_group.create_dataset("Q%.iS%.i"%(q,sz), data = Obs[qnind].T,dtype=np.double)
                # end for sz
            # end for q
        # end for i

    def Dynamics(self,ObsList,DeltaQN,GFcombi):
        # Saving the obserables
        # intput:
        #    list of list of 2D arrays - type double
        #    list of lists specifying how the Q and Sz quantum numbers change
        #    list of lists specififying how the GF are to combined.
        # saving the gfcombi internally
        self.gfcombi = GFcombi
        # ------ Observables ------
        obs_grp = self.f.create_group("Observables")
        # amount of observables
        NObs = len(ObsList)
        self.NObs = NObs
        # check if all dQNs ar given
        if NObs != len(DeltaQN):
            raise Exception('For every operator the change of quantum numbers must be specified.')
        for gfc in GFcombi:
            d1 = DeltaQN[gfc[0]]
            d2 = DeltaQN[gfc[1]]
            if d1[0] != -d2[0]:
                raise Exception('Incorrect defintion of Greens function')
            if d1[1] != -d2[1]:
                raise Exception('Incorrect defintion of Greens function')
        # amount of GFs
        NGFs = len(GFcombi)
        # ---> storing to attribute
        obs_grp.attrs['NObs'] = NObs
        obs_grp.attrs['NGFs'] = NGFs
        # saving the DeltaQN list to HDF5
        obs_grp.create_dataset("DeltaQN",data = np.array(DeltaQN,dtype='int').T,dtype='int')
        # saving the GFcombi list to HDF5
        obs_grp.create_dataset("GFcombi", data = np.array(GFcombi,dtype='int').T+1,dtype='int')
        for i in range(NObs):
            oi_group = obs_grp.create_group("Obs%.i"%(i))
            Obs = ObsList[i]
            for q in np.arange(-self.H.max_q,self.H.max_q+1):
                for sz in np.arange(-self.H.max_sz,self.H.max_sz+1):
                    # Skip QN combinations that do not contain any states
                    if not ("Q%.iS%.i"%(q,sz) in self.H.qnind): continue
                    # getting the index of the quantum number
                    qnind = self.H.qnind["Q%.iS%.i"%(q,sz)] 
                    # writing the dataset
                    dset = oi_group.create_dataset("Q%.iS%.i"%(q,sz), data = Obs[qnind].T,dtype=np.double)
                # end for sz
            # end for q
        # end for i


    def close(self):
        # switching off the conductance
        if self.cond:
            cond_group = self.f.create_group("conductance")
            cond_group.attrs['docond'] = 0
            cond_group.attrs['right'] = 0
            cond_group.attrs['left'] = 0
        # closing the HDF5 file
        self.f.flush()
        self.f.close()

    def loadtd(self,cols=(0,1)):
        return np.loadtxt(self.dir+'thermoav.dat',usecols=cols)

    def loadobs(self):
        obslist = [[ None for j in range(self.NObs)] \
                          for i in range(self.numtemps)]
        for tcount,T in enumerate(self.temps):
            for i in range(self.NObs):
                obslist[tcount][i] = np.loadtxt(self.dir+'obs_'+str(i+1)+'_T_'+("{:.2E}".format(T))+'.dat')
        return obslist

    def loadGF(self):
        gflist = [[[ None for j in range(self.NObs)] \
                          for k in range(self.NObs)] \
                          for i in range(self.numtemps)]
        tcount = 0
        for T in self.temps:
            for ind in self.gfcombi:
                gflist[tcount][ind[0]][ind[1]] = np.loadtxt(self.dir+'G_'+str(ind[0]+1)+''+str(ind[1]+1)+'_T_'+("{:.2E}".format(T))+'.dat')
            tcount += 1
        return gflist

    def loadCond(self):
        condlist = [ None for i in range(self.numtemps)]
        tcount = 0
        for T in self.temps:
            condlist[tcount] = np.loadtxt(self.dir+'C_T_'+("{:.2E}".format(T))+'.dat')
            tcount += 1
        return condlist
