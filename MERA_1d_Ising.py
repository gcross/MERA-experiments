#@+leo-ver=4-thin
#@+node:cog.20080723093036.35:@thin MERA_1d_Ising.py
#@+others
#@+node:gcross.20080724123750.3:Import needed modules
from __future__ import division

import __builtin__
import sys
sys.path.append("/Users/gcross/Projects/QC/MPS")
from numpy import *
from numpy.linalg import *
from Graph import *
from utils import *
from paulis import *
#@-node:gcross.20080724123750.3:Import needed modules
#@+node:gcross.20080728134712.3:Classes
#@+others
#@+node:gcross.20080728134712.4:Layer
class Layer(object):
    #@    @+others
    #@+node:gcross.20080728134712.5:__init__
    def __init__(self,H_terms,new_d,parent=None):
        self.H_terms = H_terms
        self.new_d = new_d
        self.parent = parent

        old_d = H_terms[0].shape[0]
        self.old_d = old_d
        U = svd(crand(old_d**2,old_d**2))[0].reshape(old_d,old_d,old_d,old_d)

        self.unitaries = [U]*4

        iso = U.reshape(old_d**2,old_d,old_d)[:new_d]
        self.isometries = [iso]*4

        self.number_of_terms = len(H_terms)
        self.number_of_unitaries = self.number_of_terms // 2
        self.number_of_isometries = self.number_of_terms // 2
    #@-node:gcross.20080728134712.5:__init__
    #@+node:gcross.20080728134712.6:optimize
    def optimize(self,number_of_iterations=30):
        for dummy in xrange(number_of_iterations):
            self.perform_optimization_iteration()
    #@-node:gcross.20080728134712.6:optimize
    #@+node:gcross.20080728134712.7:perform_optimization_iteration
    def perform_optimization_iteration(self):
        for i in xrange(self.number_of_unitaries):
            self.optimize_unitary(i)
            #if isinstance(self,Layer4): print "U", i, self.trace()

        for i in xrange(self.number_of_isometries):
            self.optimize_isometry(i)
            #if isinstance(self,Layer4): print "I", i, self.trace()
    #@-node:gcross.20080728134712.7:perform_optimization_iteration
    #@+node:gcross.20080724123750.7:optimize_unitary
    def optimize_unitary(self,index):
        UL = self.compute_UL_for_unitary(index).reshape(self.old_d**4,self.old_d**4)

    #@+at
    #     evals, evecs = eigh(UL)
    #     maxevec = evecs[:,argmax(evals)]
    #     maxevec = maxevec.reshape(self.old_d**2,self.old_d**2)
    #     u,s,v = svd(maxevec)
    #     maxevec = dot(u,v)
    # 
    #     U = maxevec.reshape((self.old_d,)*4)
    #     Ustar = U.conj()
    #@-at
    #@@c

        U = self.unitaries[index]
        Ustar = U.conj()

        for dummy in xrange(1000):
            UL_U = tensordot(UL,Ustar.ravel(),(0,0))
            W = UL_U.reshape(U.shape[0]*U.shape[1],U.shape[2]*U.shape[3]).transpose()
            W += 3*Ustar.reshape(Ustar.shape[0]*Ustar.shape[1],Ustar.shape[2]*Ustar.shape[3]).transpose()
            u,s,v = svd(W)
            U = dot(v.transpose().conj(),u.transpose().conj()).reshape(U.shape)
            #Ustar = U.conj()

            UL_Ustar = tensordot(UL,U.ravel(),(1,0))
            W = UL_Ustar.reshape(Ustar.shape[0]*Ustar.shape[1],Ustar.shape[2]*Ustar.shape[3]).transpose()
            W += 3*U.reshape(U.shape[0]*U.shape[1],U.shape[2]*U.shape[3]).transpose()
            u,s,v = svd(W)
            Ustar = dot(v.transpose().conj(),u.transpose().conj()).reshape(Ustar.shape)

            if norm(U.conj()-Ustar)<1e-8: break

            #if dummy%10==0: print "\t",dummy, norm(U.conj()-Ustar)

        self.unitaries[index] = U
    #@-node:gcross.20080724123750.7:optimize_unitary
    #@+node:cog.20080724235649.2:optimize_isometry
    def optimize_isometry(self,index):
        UL = self.compute_UL_for_isometry(index).reshape(self.old_d**2,self.old_d**2)

        evals, evecs = eigh(UL)

        indices = argsort(evals)[-self.new_d:]

        self.isometries[index] = evecs.transpose()[indices].reshape(self.new_d,self.old_d,self.old_d)

    #@-node:cog.20080724235649.2:optimize_isometry
    #@+node:cog.20080724235649.7:trace
    def trace(self):
        I = identity(self.old_d)
        tr = 0
        for index in xrange(self.number_of_terms//2):
            tr += self.fully_contract_terms_at(index)
        return tr

    #@-node:cog.20080724235649.7:trace
    #@+node:cog.20080724235649.8:build_full_hamiltonian
    def build_full_hamiltonian(self):
        number_of_terms = len(self.H_terms)
        H = zeros((self.old_d,self.old_d)*number_of_terms,complex128)
        I = identity(self.old_d)
        for i, term in enumerate(self.H_terms):
            term = term.transpose(0,3,1,4,2,5)
            if i == number_of_terms-2:
                new_indices = range(2*(number_of_terms-1),2*number_of_terms) + range(2*(number_of_terms-1))
                contribution = reduce(multiply.outer,[I]*(number_of_terms-3)+[term]).transpose(new_indices)
            elif i == number_of_terms-1:
                new_indices = range(2*(number_of_terms-2),2*number_of_terms) + range(2*(number_of_terms-2))
                contribution = reduce(multiply.outer,[I]*(number_of_terms-3)+[term]).transpose(new_indices)
            else:
                contribution = reduce(multiply.outer,[I]*i+[term]+[I]*(number_of_terms-(3+i)))
            assert H.shape == contribution.shape    
            H += contribution
        return H.transpose(range(0,2*number_of_terms,2)+range(1,2*number_of_terms,2)).reshape((self.old_d**number_of_terms,)*2)
    #@-node:cog.20080724235649.8:build_full_hamiltonian
    #@+node:cog.20080724235649.10:renormalize
    def renormalize(self,new_d=None):
        new_H_terms = []
        for index in range(1,len(self.H_terms)//2)+[0]:
            new_H_terms.append(self.renormalize_term(index))        
        if new_d == None:
            new_d = self.new_d
        if len(new_H_terms) > 4:
            return LayerG4(new_H_terms,new_d,self)
        elif len(new_H_terms) == 4:
            return Layer4(new_H_terms,new_d,self)
        else:
            assert len(new_H_terms) == 2
            return TopLayer(new_H_terms,self)
    #@-node:cog.20080724235649.10:renormalize
    #@+node:gcross.20080728134712.8:spectrum
    def spectrum(self):
        return eigvalsh(self.build_full_hamiltonian())
    #@-node:gcross.20080728134712.8:spectrum
    #@-others
#@-node:gcross.20080728134712.4:Layer
#@+node:gcross.20080728134712.14:LayerG4
class LayerG4(Layer):
    #@    @+others
    #@+node:gcross.20080728134712.17:Contractors
    #@+others
    #@+node:gcross.20080728134712.18:UL conversions
    g_left = make_placeholder_graph(
        ("IA",(2,2,2),(0,1,2)),
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(2,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IA*",(2,2,2),(0,11,12)),
        ("IB*",(2,2,2),(3,14,15)),
        ("IC*",(2,2,2),(6,17,18)),
        ("UA*",(2,2,2,2),(12,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(31,-32,33,21,-22,23)),
        ("IAI",(2,2),(1,11)),
        ("ICI",(2,2),(8,18)),
        ("UAI",(2,2),(22,-22)),
        ("UAI*",(2,2),(32,-32)),
        ("UBI",(2,2),(24,34)),
    )

    make_UL_IA_left = staticmethod(compile_graph(g_left,[1,2,3,4,6,7,8,9,10,11,12,13,14,15],["IAI","ICI","UAI","UBI","IB","IC","UA","UB","H"],node_ordering=[5,0]))
    make_UL_IB_left = staticmethod(compile_graph(g_left,[0,2,3,4,5,7,8,9,10,11,12,13,14,15],["IAI","ICI","UAI","UBI","IA","IC","UA","UB","H"],node_ordering=[6,1]))
    make_UL_IC_left = staticmethod(compile_graph(g_left,[0,1,3,4,5,6,8,9,10,11,12,13,14,15],["IAI","ICI","UAI","UBI","IA","IB","UA","UB","H"],node_ordering=[7,2]))
    make_UL_UA_left = staticmethod(compile_graph(g_left,[0,1,2,4,5,6,7,9,10,11,12,13,14,15],["IAI","ICI","UAI","UBI","IA","IB","IC","UB","H"],node_ordering=[8,3]))
    make_UL_UB_left = staticmethod(compile_graph(g_left,[0,1,2,3,5,6,7,8,10,11,12,13,14,15],["IAI","ICI","UAI","UBI","IA","IB","IC","UA","H"],node_ordering=[9,4]))
    fully_contract_left = staticmethod(compile_graph(g_left,range(16),["IAI","ICI","UAI","UBI","IA","IB","IC","UA","UB","H"],node_ordering=[]))

    g_right = make_placeholder_graph(
        ("IA",(2,2,2),(0,1,2)),
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(2,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IA*",(2,2,2),(0,11,12)),
        ("IB*",(2,2,2),(3,14,15)),
        ("IC*",(2,2,2),(6,17,18)),
        ("UA*",(2,2,2,2),(12,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(32,33,-34,22,23,-24)),
        ("IAI",(2,2),(1,11)),
        ("ICI",(2,2),(8,18)),
        ("UCI",(2,2),(24,-24)),
        ("UCI*",(2,2),(34,-34)),
        ("UBI",(2,2),(21,31)),
    )

    make_UL_IA_right = staticmethod(compile_graph(g_right,[1,2,3,4,6,7,8,9,10,11,12,13,14,15],["IAI","ICI","UBI","UCI","IB","IC","UA","UB","H"],node_ordering=[5,0]))
    make_UL_IB_right = staticmethod(compile_graph(g_right,[0,2,3,4,5,7,8,9,10,11,12,13,14,15],["IAI","ICI","UBI","UCI","IA","IC","UA","UB","H"],node_ordering=[6,1]))
    make_UL_IC_right = staticmethod(compile_graph(g_right,[0,1,3,4,5,6,8,9,10,11,12,13,14,15],["IAI","ICI","UBI","UCI","IA","IB","UA","UB","H"],node_ordering=[7,2]))
    make_UL_UA_right = staticmethod(compile_graph(g_right,[0,1,2,4,5,6,7,9,10,11,12,13,14,15],["IAI","ICI","UBI","UCI","IA","IB","IC","UB","H"],node_ordering=[8,3]))
    make_UL_UB_right = staticmethod(compile_graph(g_right,[0,1,2,3,5,6,7,8,10,11,12,13,14,15],["IAI","ICI","UBI","UCI","IA","IB","IC","UA","H"],node_ordering=[9,4]))
    fully_contract_right = staticmethod(compile_graph(g_right,range(16),["IAI","ICI","UBI","UCI","IA","IB","IC","UA","UB","H"],node_ordering=[]))
    #@-node:gcross.20080728134712.18:UL conversions
    #@+node:gcross.20080728134712.19:Renormalization tranformations
    _renormalize_right_H = staticmethod(make_contractor_from_implicit_joins([
        (32,33,34,22,23,24), # H
        (5,7,23,24), # UB
        (15,17,33,34), # UB*
        (2,4,21,22), # UA
        (12,14,21,32), # UA*
        (3,4,5), # IB
        (13,14,15), # IB*
        (0,1,2), # IA
        (10,1,12), # IA*
        (6,7,8), # IC
        (16,17,8), # IC*
    ],[
        10,
        13,
        16,
        0,
        3,
        6,
    ]))


    renormalize_right_H = lambda self,IA,IB,IC,UA,UB,H: \
        self._renormalize_right_H(H,UB,UB.conj(),UA,UA.conj(),IB,IB.conj(),IA,IA.conj(),IC,IC.conj())

    _renormalize_left_H = staticmethod(make_contractor_from_implicit_joins([
        (31,32,33,21,22,23), # H
        (2,4,21,22), # UA
        (12,14,31,32), # UA*
        (5,7,23,24), # UB
        (15,17,33,24), # UB*
        (3,4,5), # IB
        (13,14,15), # IB*
        (0,1,2), # IA
        (10,1,12), # IA*
        (6,7,8), # IC
        (16,17,8), # IC*
    ],[
        10,
        13,
        16,
        0,
        3,
        6,
    ]))

    renormalize_left_H = lambda self,IA,IB,IC,UA,UB,H: \
        self._renormalize_left_H(H,UA,UA.conj(),UB,UB.conj(),IB,IB.conj(),IA,IA.conj(),IC,IC.conj())
    #@-node:gcross.20080728134712.19:Renormalization tranformations
    #@-others
    #@-node:gcross.20080728134712.17:Contractors
    #@+node:gcross.20080728134712.26:compute_UL_for_unitary
    def compute_UL_for_unitary(self,index):
        I = identity(self.old_d)

        left_unitary = self.unitaries[(index-1) % self.number_of_unitaries]
        unitary = self.unitaries[index]
        right_unitary = self.unitaries[(index+1) % self.number_of_unitaries]

        next_left_isometry = self.isometries[(index-2) % self.number_of_isometries]
        left_isometry = self.isometries[(index-1) % self.number_of_isometries]
        right_isometry = self.isometries[(index+0) % self.number_of_isometries]
        next_right_isometry = self.isometries[(index+1) % self.number_of_isometries]

        next_left_UL = self.make_UL_UB_left(
            I,I,I,I,
            next_left_isometry,
            left_isometry,
            right_isometry,
            left_unitary,
            self.H_terms[(2*index-2) % self.number_of_terms]
        )

        left_UL = self.make_UL_UB_right(
            I,I,I,I,
            next_left_isometry,
            left_isometry,
            right_isometry,
            left_unitary,
            self.H_terms[(2*index-1) % self.number_of_terms]
        )

        right_UL = self.make_UL_UA_left(
            I,I,I,I,
            left_isometry,
            right_isometry,
            next_right_isometry,
            right_unitary,
            self.H_terms[(2*index+0) % self.number_of_terms]
        )

        next_right_UL = self.make_UL_UA_right(
            I,I,I,I,
            left_isometry,
            right_isometry,
            next_right_isometry,
            right_unitary,
            self.H_terms[(2*index+1) % self.number_of_terms]
        )

        return next_left_UL+left_UL+right_UL+next_right_UL
    #@-node:gcross.20080728134712.26:compute_UL_for_unitary
    #@+node:gcross.20080728134712.40:compute_UL_for_isometry
    def compute_UL_for_isometry(self,index):
        I = identity(self.old_d)

        next_left_unitary = self.unitaries[(index-1) % self.number_of_unitaries]
        left_unitary = self.unitaries[(index-0) % self.number_of_unitaries]
        right_unitary = self.unitaries[(index+1) % self.number_of_unitaries]
        next_right_unitary = self.unitaries[(index+2) % self.number_of_unitaries]

        next_left_isometry = self.isometries[(index-2) % self.number_of_isometries]
        left_isometry = self.isometries[(index-1) % self.number_of_isometries]
        isometry = self.isometries[index]
        right_isometry = self.isometries[(index+1) % self.number_of_isometries]
        next_right_isometry = self.isometries[(index+2) % self.number_of_isometries]

        LLL_UL = self.make_UL_IC_left(
            I,I,I,I,
            next_left_isometry,
            left_isometry,
            next_left_unitary,
            left_unitary,
            self.H_terms[(2*index-2) % self.number_of_terms]
        )

        LL_UL = self.make_UL_IC_right(
            I,I,I,I,
            next_left_isometry,
            left_isometry,
            next_left_unitary,
            left_unitary,
            self.H_terms[(2*index-1) % self.number_of_terms]
        )

        L_UL = self.make_UL_IB_left(
            I,I,I,I,
            left_isometry,
            right_isometry,
            left_unitary,
            right_unitary,
            self.H_terms[(2*index-0) % self.number_of_terms]
        )

        R_UL = self.make_UL_IB_right(
            I,I,I,I,
            left_isometry,
            right_isometry,
            left_unitary,
            right_unitary,
            self.H_terms[(2*index+1) % self.number_of_terms]
        )

        RR_UL = self.make_UL_IA_left(
            I,I,I,I,
            right_isometry,
            next_right_isometry,
            right_unitary,
            next_right_unitary,
            self.H_terms[(2*index+2) % self.number_of_terms]
        )

        RRR_UL = self.make_UL_IA_right(
            I,I,I,I,
            right_isometry,
            next_right_isometry,
            right_unitary,
            next_right_unitary,
            self.H_terms[(2*index+3) % self.number_of_terms]
        )

        return LLL_UL+LL_UL+L_UL+R_UL+RR_UL+RRR_UL

    #@-node:gcross.20080728134712.40:compute_UL_for_isometry
    #@+node:gcross.20080728134712.31:fully_contract_terms_at
    def fully_contract_terms_at(index):
        I = identity(self.old_d)
        return self.fully_contract_left(
            I,I,I,I,
            self.isometries[(index-1) % self.number_of_isometries],
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index+0]
        ) + self.fully_contract_right(
            I,I,I,I,
            self.isometries[(index-1) % self.number_of_isometries],
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index+1]
        )
    #@-node:gcross.20080728134712.31:fully_contract_terms_at
    #@+node:gcross.20080728134712.33:renormalize_term
    def renormalize_term(self,index):
        return self.renormalize_left_H(
            self.isometries[(index-1) % self.number_of_isometries],
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index]
        ) + self.renormalize_right_H(
            self.isometries[(index-1) % self.number_of_isometries],
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index+1]
        )
    #@-node:gcross.20080728134712.33:renormalize_term
    #@-others
#@-node:gcross.20080728134712.14:LayerG4
#@+node:gcross.20080728134712.35:Layer4
class Layer4(Layer):
    #@    @+others
    #@+node:gcross.20080728134712.11:Contractors
    #@+others
    #@+node:gcross.20080728134712.12:UL conversions
    g_left = make_placeholder_graph(
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(8,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IB*",(2,2,2),(3,14,15)),
        ("IC*",(2,2,2),(6,17,18)),
        ("UA*",(2,2,2,2),(18,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(31,-32,33,21,-22,23)),
        ("UAI",(2,2),(22,-22)),
        ("UAI*",(2,2),(32,-32)),
        ("UBI",(2,2),(24,34)),
    )

    make_UL_IB_left = staticmethod(compile_graph(g_left,range(0,0) + range(1,4) + range(5,12),["UAI","UBI","IC","UA","UB","H"],node_ordering=[4,0]))
    make_UL_IC_left = staticmethod(compile_graph(g_left,range(0,1) + range(2,5) + range(6,12),["UAI","UBI","IB","UA","UB","H"],node_ordering=[5,1]))
    make_UL_UA_left = staticmethod(compile_graph(g_left,range(0,2) + range(3,6) + range(7,12),["UAI","UBI","IB","IC","UB","H"],node_ordering=[6,2]))
    make_UL_UB_left = staticmethod(compile_graph(g_left,range(0,3) + range(4,7) + range(8,12),["UAI","UBI","IB","IC","UA","H"],node_ordering=[7,3]))
    fully_contract_left = staticmethod(compile_graph(g_left,range(12),["UAI","UBI","IB","IC","UA","UB","H"],node_ordering=[]))

    g_right = make_placeholder_graph(
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(8,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IB*",(2,2,2),(3,14,15)),
        ("IC*",(2,2,2),(6,17,18)),
        ("UA*",(2,2,2,2),(18,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(32,33,-34,22,23,-24)),
        ("UBI",(2,2),(24,-24)),
        ("UBI*",(2,2),(34,-34)),
        ("UAI",(2,2),(21,31)),
    )

    make_UL_IB_right = staticmethod(compile_graph(g_right,range(0,0) + range(1,4) + range(5,12),["UAI","UBI","IC","UA","UB","H"],node_ordering=[4,0]))
    make_UL_IC_right = staticmethod(compile_graph(g_right,range(0,1) + range(2,5) + range(6,12),["UAI","UBI","IB","UA","UB","H"],node_ordering=[5,1]))
    make_UL_UA_right = staticmethod(compile_graph(g_right,range(0,2) + range(3,6) + range(7,12),["UAI","UBI","IB","IC","UB","H"],node_ordering=[6,2]))
    make_UL_UB_right = staticmethod(compile_graph(g_right,range(0,3) + range(4,7) + range(8,12),["UAI","UBI","IB","IC","UA","H"],node_ordering=[7,3]))
    fully_contract_right = staticmethod(compile_graph(g_right,range(12),["UAI","UBI","IB","IC","UA","UB","H"],node_ordering=[]))
    #@-node:gcross.20080728134712.12:UL conversions
    #@+node:gcross.20080728134712.13:Renormalization tranformations
    _renormalize_right_H = staticmethod(make_contractor_from_implicit_joins([
        (32,33,34,22,23,24), # H
        (5,7,23,24), # UB
        (15,17,33,34), # UB*
        (8,4,21,22), # UA
        (18,14,21,32), # UA*
        (3,4,5), # IB
        (13,14,15), # IB*
        (6,7,8), # IC
        (16,17,18), # IC*
    ],[
        13,
        16,
        3,
        6,
    ]))


    renormalize_right_H = lambda self,IB,IC,UA,UB,H: \
        self._renormalize_right_H(H,UB,UB.conj(),UA,UA.conj(),IB,IB.conj(),IC,IC.conj())

    _renormalize_left_H = staticmethod(make_contractor_from_implicit_joins([
        (31,32,33,21,22,23), # H
        (8,4,21,22), # UA
        (18,14,31,32), # UA*
        (5,7,23,24), # UB
        (15,17,33,24), # UB*
        (3,4,5), # IB
        (13,14,15), # IB*
        (6,7,8), # IC
        (16,17,18), # IC*
    ],[
        13,
        16,
        3,
        6,
    ]))

    renormalize_left_H = lambda self,IB,IC,UA,UB,H: \
        self._renormalize_left_H(H,UA,UA.conj(),UB,UB.conj(),IB,IB.conj(),IC,IC.conj())
    #@-node:gcross.20080728134712.13:Renormalization tranformations
    #@-others
    #@-node:gcross.20080728134712.11:Contractors
    #@+node:gcross.20080728134712.39:compute_UL_for_unitary
    def compute_UL_for_unitary(self,index):
        I = identity(self.old_d)

        other_unitary = self.unitaries[1-index]

        left_isometry = self.isometries[(index-1) % self.number_of_isometries]
        right_isometry = self.isometries[(index+0) % self.number_of_isometries]

        next_left_UL = self.make_UL_UB_left(
            I,I,
            left_isometry,
            right_isometry,
            other_unitary,
            self.H_terms[(2*index-2) % self.number_of_terms]
        )

        left_UL = self.make_UL_UB_right(
            I,I,
            left_isometry,
            right_isometry,
            other_unitary,
            self.H_terms[(2*index-1) % self.number_of_terms]
        )

        right_UL = self.make_UL_UA_left(
            I,I,
            right_isometry,
            left_isometry,
            other_unitary,
            self.H_terms[(2*index+0) % self.number_of_terms]
        )

        next_right_UL = self.make_UL_UA_right(
            I,I,
            right_isometry,
            left_isometry,
            other_unitary,
            self.H_terms[(2*index+1) % self.number_of_terms]
        )

        return next_left_UL+left_UL+right_UL+next_right_UL
    #@-node:gcross.20080728134712.39:compute_UL_for_unitary
    #@+node:gcross.20080728134712.29:compute_UL_for_isometry
    def compute_UL_for_isometry(self,index):
        I = identity(self.old_d)

        other_isometry = self.isometries[1-index]

        left_unitary = self.unitaries[(index-0) % self.number_of_unitaries]
        right_unitary = self.unitaries[(index+1) % self.number_of_unitaries]

        next_left_UL = self.make_UL_IC_left(
            I,I,
            other_isometry,
            right_unitary,
            left_unitary,
            self.H_terms[(2*index-2) % self.number_of_terms]
        )

        left_UL = self.make_UL_IC_right(
            I,I,
            other_isometry,
            right_unitary,
            left_unitary,
            self.H_terms[(2*index-1) % self.number_of_terms]
        )

        right_UL = self.make_UL_IB_left(
            I,I,
            other_isometry,
            left_unitary,
            right_unitary,
            self.H_terms[(2*index+0) % self.number_of_terms]
        )

        next_right_UL = self.make_UL_IB_right(
            I,I,
            other_isometry,
            left_unitary,
            right_unitary,
            self.H_terms[(2*index+1) % self.number_of_terms]
        )

        return next_left_UL+left_UL+right_UL+next_right_UL

    #@-node:gcross.20080728134712.29:compute_UL_for_isometry
    #@+node:gcross.20080728134712.41:fully_contract_terms_at
    def fully_contract_terms_at(self,index):
        I = identity(self.old_d)
        return self.fully_contract_left(
            I,I,
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index+0]
        ) + self.fully_contract_right(
            I,I,
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index+1]
        )
    #@-node:gcross.20080728134712.41:fully_contract_terms_at
    #@+node:gcross.20080728134712.42:renormalize_term
    def renormalize_term(self,index):
        return self.renormalize_left_H(
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index]
        ) + self.renormalize_right_H(
            self.isometries[(index+0) % self.number_of_isometries],
            self.isometries[(index+1) % self.number_of_isometries],
            self.unitaries[(index+0) % self.number_of_unitaries],
            self.unitaries[(index+1) % self.number_of_unitaries],
            self.H_terms[2*index+1]
        )
    #@-node:gcross.20080728134712.42:renormalize_term
    #@-others
#@-node:gcross.20080728134712.35:Layer4
#@+node:gcross.20080728134712.44:TopLayer
class TopLayer(object):
    #@    @+others
    #@+node:gcross.20080728134712.45:__init__
    def __init__(self,H_terms,parent):
        self.parent = parent
        H = H_terms[0] + H_terms[1].transpose(1,0,3,2)
        H = H.reshape(H.shape[0]*H.shape[1],H.shape[2]*H.shape[3])
        self.H = H
        self.H_terms = H_terms


    #@-node:gcross.20080728134712.45:__init__
    #@+node:gcross.20080728134712.47:trace
    def trace(self):
        return trace(self.H)


    #@-node:gcross.20080728134712.47:trace
    #@+node:gcross.20080728134712.46:spectrum
    def spectrum(self):
        return eigvalsh(self.H)
    #@-node:gcross.20080728134712.46:spectrum
    #@-others
#@-node:gcross.20080728134712.44:TopLayer
#@-others
#@-node:gcross.20080728134712.3:Classes
#@+node:gcross.20080728134712.9:Initialization
term = (reduce(multiply.outer,[0.5*X,X,I])+reduce(multiply.outer,[Z,I,I])).transpose(0,2,4,1,3,5)

H_terms = [term]*8

bottom_layer = LayerG4(H_terms,3)
#@-node:gcross.20080728134712.9:Initialization
#@+node:cog.20080724235649.5:Main Loop
for dummy in xrange(10):
    bottom_layer.optimize(10)
    print dummy, bottom_layer.renormalize(2).spectrum()

next_layer = bottom_layer.renormalize(3)
print next_layer.spectrum()

for dummy in xrange(100):
    next_layer.optimize(1)
    #next_next_layer = next_layer.renormalize()
    if dummy % 10 == 0:
        print dummy, next_layer.trace()
    #print "\t",eigvalsh(next_next_layer.H_terms[0].reshape(9,9))
    #print "\t",eigvalsh(next_next_layer.H_terms[1].reshape(9,9))

next_next_layer = next_layer.renormalize()
print next_next_layer.spectrum()

for U in bottom_layer.unitaries[1:]:
    print norm(bottom_layer.unitaries[0]-U)
for U in next_layer.unitaries[1:]:
    print norm(next_layer.unitaries[0]-U)
for iso in bottom_layer.isometries[1:]:
    print norm(bottom_layer.isometries[0]-iso)
for iso in next_layer.isometries[1:]:
    print norm(next_layer.isometries[0]-iso)
#@-node:cog.20080724235649.5:Main Loop
#@-others
#@-node:cog.20080723093036.35:@thin MERA_1d_Ising.py
#@-leo
