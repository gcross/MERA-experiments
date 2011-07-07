#@+leo-ver=4-thin
#@+node:cog.20080728182049.3:@thin MERA_1d_Ising_TI.py
#@+others
#@+node:cog.20080728182049.4:Import needed modules
from __future__ import division

import __builtin__
import sys
sys.path.append("/Users/gcross/Projects/QC/MPS")
from numpy import *
from numpy.linalg import *
from Graph import *
from utils import *
from paulis import *
#@-node:cog.20080728182049.4:Import needed modules
#@+node:gcross.20080730130632.3:Functions
#@+others
#@+node:gcross.20080730130632.4:optimize_UQ
def optimize_UQ(UL,guess,left_d,right_d):
    UL = UL.reshape(left_d*right_d,left_d*right_d)

    U = guess.reshape(left_d,right_d)
    Ustar = U.conj()

    for dummy in xrange(1000):
        UL_U = tensordot(UL,Ustar.ravel(),(0,0))
        W = UL_U.reshape(left_d,right_d).transpose()
        W += 10*Ustar.reshape(left_d,right_d).transpose()
        u,s,v = svd(W.transpose().conj(),full_matrices=0)
        U = dot(u,v).reshape(U.shape)
        #Ustar = U.conj()

        UL_Ustar = tensordot(UL,U.ravel(),(1,0))
        W = UL_Ustar.reshape(left_d,right_d).transpose()
        W += 10*U.reshape(left_d,right_d).transpose()
        u,s,v = svd(W.transpose().conj(),full_matrices=0)
        Ustar = dot(u,v).reshape(Ustar.shape)

        if norm(U.conj()-Ustar)<1e-5: break

        #if dummy%200==0: print "\t",dummy, norm(U.conj()-Ustar)
        if dummy > 100 and norm(U.conj()-Ustar)>1:
            break

    #print dummy, norm(U.conj()-Ustar)

    return U
#@-node:gcross.20080730130632.4:optimize_UQ
#@-others
#@-node:gcross.20080730130632.3:Functions
#@+node:cog.20080728182049.5:Classes
#@+others
#@+node:cog.20080728182049.6:Layer
class Layer(object):
    #@    @+others
    #@+node:cog.20080728182049.7:__init__
    def __init__(self,H_term,number_of_sites,new_d,parent=None):
        self.set_H_term(H_term)
        self.new_d = new_d
        self.parent = parent
        self.number_of_sites = number_of_sites

    #@+at
    #     old_d = H_term.shape[0]
    #     self.old_d = old_d
    #     U = 
    # svd(crand(old_d**2,old_d**2))[0].reshape(old_d,old_d,old_d,old_d)
    # 
    #     self.unitary = U
    #     self.isometry = U.reshape(old_d**2,old_d,old_d)[:new_d]
    #@-at
    #@@c

        old_d = H_term.shape[0]
        self.old_d = old_d
        X = crand(old_d,old_d,old_d,old_d)
        X = X + X.transpose(1,0,3,2)
        u,s,v = svd(X.reshape(old_d**2,old_d**2))
        W = dot(u,v)
        assert norm(dot(W,W.conj().transpose())-identity(old_d**2)) < 1e-13
        W = W.reshape(old_d,old_d,old_d,old_d)
        assert norm(W-W.transpose(1,0,3,2)) < 1e-13
        self.unitary = W

        X = crand(new_d,old_d,old_d)
        X = X + X.transpose(0,2,1)
        u,s,v = svd(X.reshape(new_d,old_d**2),full_matrices=False)
        W = dot(u,v)
        assert norm(dot(W,W.conj().transpose())-identity(new_d)) < 1e-13
        W = W.reshape(new_d,old_d,old_d)
        assert norm(W-W.transpose(0,2,1)) < 1e-13
        self.isometry = W
    #@-node:cog.20080728182049.7:__init__
    #@+node:gcross.20080730142552.3:set_H_term
    def set_H_term(self,H_term):
        self.H_term = H_term
    #@-node:gcross.20080730142552.3:set_H_term
    #@+node:cog.20080728182049.8:optimize
    def optimize(self,number_of_iterations=30):
        for dummy in xrange(number_of_iterations):
            self.perform_optimization_iteration()
    #@-node:cog.20080728182049.8:optimize
    #@+node:cog.20080728182049.9:perform_optimization_iteration
    def perform_optimization_iteration(self):
        self.optimize_unitary()
        self.optimize_isometry()
    #@-node:cog.20080728182049.9:perform_optimization_iteration
    #@+node:cog.20080728182049.10:optimize_unitary
    def optimize_unitary(self):
        UL = self.compute_UL_for_unitary()

        self.unitary = optimize_UQ(UL,self.unitary,self.old_d**2,self.old_d**2).reshape(self.unitary.shape)

    #@+at
    #     U = 
    # optimize_UQ(UL,self.unitary,self.old_d**2,self.old_d**2).reshape(self.unitary.shape)
    # 
    #     W = U+U.transpose(1,0,3,2)
    #     W = W.reshape(W.shape[0]*W.shape[1],W.shape[2]*W.shape[3])
    #     u,s,v = svd(W,full_matrices=0)
    #     W = dot(u,v).reshape(self.unitary.shape)
    #     self.unitary = W
    # 
    #@-at
    #@@c
    #@-node:cog.20080728182049.10:optimize_unitary
    #@+node:cog.20080728182049.11:optimize_isometry
    def optimize_isometry(self):
        UL = self.compute_UL_for_isometry()

        self.isometry = optimize_UQ(UL,self.isometry,self.new_d,self.old_d**2).reshape(self.isometry.shape)
    #@-node:cog.20080728182049.11:optimize_isometry
    #@+node:cog.20080728182049.12:trace
    def trace(self):
        return self.number_of_sites*self.fully_contract_term()
    #@-node:cog.20080728182049.12:trace
    #@+node:cog.20080728182049.13:build_full_hamiltonian
    def build_full_hamiltonian(self):
        number_of_sites = self.number_of_sites
        H = zeros((self.old_d,self.old_d)*number_of_sites,complex128)
        I = identity(self.old_d)
        term = self.H_term.transpose(0,3,1,4,2,5)
        for i in xrange(number_of_sites):        
            if i == number_of_sites-2:
                new_indices = range(2*(number_of_sites-1),2*number_of_sites) + range(2*(number_of_sites-1))
                contribution = reduce(multiply.outer,[I]*(number_of_sites-3)+[term]).transpose(new_indices)
            elif i == number_of_sites-1:
                new_indices = range(2*(number_of_sites-2),2*number_of_sites) + range(2*(number_of_sites-2))
                contribution = reduce(multiply.outer,[I]*(number_of_sites-3)+[term]).transpose(new_indices)
            else:
                contribution = reduce(multiply.outer,[I]*i+[term]+[I]*(number_of_sites-(3+i)))
            assert H.shape == contribution.shape    
            H += contribution
        return H.transpose(range(0,2*number_of_sites,2)+range(1,2*number_of_sites,2)).reshape((self.old_d**number_of_sites,)*2)
    #@-node:cog.20080728182049.13:build_full_hamiltonian
    #@+node:cog.20080728182049.14:create_renormalized_layer
    def create_renormalized_layer(self,new_d=None):
        new_H_term = self.renormalize_term()
        if new_d == None:
            new_d = self.new_d
        if self.number_of_sites > 8:
            self.child = LayerG4(new_H_term,self.number_of_sites//2,new_d,self)
        elif self.number_of_sites == 8:
            self.child = Layer4(new_H_term,self.number_of_sites//2,new_d,self)
        else:
            assert self.number_of_sites == 4
            self.child = TopLayer(new_H_term,new_d,self)
        return self.child

    #@-node:cog.20080728182049.14:create_renormalized_layer
    #@+node:gcross.20080730130632.9:renormalize
    def renormalize(self):
        self.child.set_H_term(self.renormalize_term())

    #@-node:gcross.20080730130632.9:renormalize
    #@+node:gcross.20080730130632.7:project
    def project(self):
        self.parent.projector = self.compute_projector()
    #@-node:gcross.20080730130632.7:project
    #@+node:cog.20080728182049.15:spectrum
    def spectrum(self):
        return eigvalsh(self.build_full_hamiltonian())
    #@-node:cog.20080728182049.15:spectrum
    #@-others
#@-node:cog.20080728182049.6:Layer
#@+node:cog.20080728182049.16:LayerG4
class LayerG4(Layer):
    #@    @+others
    #@+node:gcross.20080730130632.2:__init__
    def __init__(self,*args,**keywords):
        super(self.__class__,self).__init__(*args,**keywords)
        self.projector = identity(self.new_d**3).reshape((self.new_d,)*6)
    #@-node:gcross.20080730130632.2:__init__
    #@+node:cog.20080728182049.17:Contractors
    #@+others
    #@+node:cog.20080728182049.18:UL conversions
    g_left = make_placeholder_graph(
        ("IA",(2,2,2),(0,1,2)),
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(2,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IA*",(2,2,2),(10,11,12)),
        ("IB*",(2,2,2),(13,14,15)),
        ("IC*",(2,2,2),(16,17,18)),
        ("UA*",(2,2,2,2),(12,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(31,-32,33,21,-22,23)),
        ("P",(2,2,2,2,2,2),(10,13,16,0,3,6)),
        ("IAI",(2,2),(1,11)),
        ("ICI",(2,2),(8,18)),
        ("UAI",(2,2),(22,-22)),
        ("UAI*",(2,2),(32,-32)),
        ("UBI",(2,2),(24,34)),
    )

    make_UL_IA_left = staticmethod(compile_graph(g_left,range(0,0)+range(1,5)+range( 6,17),["IAI","ICI","UAI","UBI","IB","IC","UA","UB","H","P"],node_ordering=[5,0]))
    make_UL_IB_left = staticmethod(compile_graph(g_left,range(0,1)+range(2,6)+range( 7,17),["IAI","ICI","UAI","UBI","IA","IC","UA","UB","H","P"],node_ordering=[6,1]))
    make_UL_IC_left = staticmethod(compile_graph(g_left,range(0,2)+range(3,7)+range( 8,17),["IAI","ICI","UAI","UBI","IA","IB","UA","UB","H","P"],node_ordering=[7,2]))
    make_UL_UA_left = staticmethod(compile_graph(g_left,range(0,3)+range(4,8)+range( 9,17),["IAI","ICI","UAI","UBI","IA","IB","IC","UB","H","P"],node_ordering=[8,3]))
    make_UL_UB_left = staticmethod(compile_graph(g_left,range(0,4)+range(5,9)+range(10,17),["IAI","ICI","UAI","UBI","IA","IB","IC","UA","H","P"],node_ordering=[9,4]))
    fully_contract_left = staticmethod(compile_graph(g_left,range(17),["IAI","ICI","UAI","UBI","IA","IB","IC","UA","UB","H","P"],node_ordering=[]))
    make_left_projector = staticmethod(compile_graph(g_left,range(10)+range(11,17),["IAI","ICI","UAI","UBI","IA","IB","IC","UA","UB","P"],node_ordering=[10]))

    g_right = make_placeholder_graph(
        ("IA",(2,2,2),(0,1,2)),
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(2,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IA*",(2,2,2),(10,11,12)),
        ("IB*",(2,2,2),(13,14,15)),
        ("IC*",(2,2,2),(16,17,18)),
        ("UA*",(2,2,2,2),(12,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(32,33,-34,22,23,-24)),
        ("P",(2,2,2,2,2,2),(10,13,16,0,3,6)),
        ("IAI",(2,2),(1,11)),
        ("ICI",(2,2),(8,18)),
        ("UCI",(2,2),(24,-24)),
        ("UCI*",(2,2),(34,-34)),
        ("UBI",(2,2),(21,31)),
    )

    make_UL_IA_right = staticmethod(compile_graph(g_right,range(0,0)+range(1,5)+range( 6,17),["IAI","ICI","UBI","UCI","IB","IC","UA","UB","H","P"],node_ordering=[5,0]))
    make_UL_IB_right = staticmethod(compile_graph(g_right,range(0,1)+range(2,6)+range( 7,17),["IAI","ICI","UBI","UCI","IA","IC","UA","UB","H","P"],node_ordering=[6,1]))
    make_UL_IC_right = staticmethod(compile_graph(g_right,range(0,2)+range(3,7)+range( 8,17),["IAI","ICI","UBI","UCI","IA","IB","UA","UB","H","P"],node_ordering=[7,2]))
    make_UL_UA_right = staticmethod(compile_graph(g_right,range(0,3)+range(4,8)+range( 9,17),["IAI","ICI","UBI","UCI","IA","IB","IC","UB","H","P"],node_ordering=[8,3]))
    make_UL_UB_right = staticmethod(compile_graph(g_right,range(0,4)+range(5,9)+range(10,17),["IAI","ICI","UBI","UCI","IA","IB","IC","UA","H","P"],node_ordering=[9,4]))
    fully_contract_right = staticmethod(compile_graph(g_right,range(17),["IAI","ICI","UBI","UCI","IA","IB","IC","UA","UB","H","P"],node_ordering=[]))
    make_right_projector = staticmethod(compile_graph(g_right,range(10)+range(11,17),["IAI","ICI","UBI","UCI","IA","IB","IC","UA","UB","P"],node_ordering=[10]))
    #@-node:cog.20080728182049.18:UL conversions
    #@+node:cog.20080728182049.19:Renormalization tranformations
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
    #@-node:cog.20080728182049.19:Renormalization tranformations
    #@-others
    #@-node:cog.20080728182049.17:Contractors
    #@+node:cog.20080728182049.20:compute_UL_for_unitary
    def compute_UL_for_unitary(self):
        I = identity(self.old_d)

        next_left_UL = self.make_UL_UB_left(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        left_UL = self.make_UL_UB_right(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        right_UL = self.make_UL_UA_left(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        next_right_UL = self.make_UL_UA_right(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        return next_left_UL+left_UL+right_UL+next_right_UL
    #@-node:cog.20080728182049.20:compute_UL_for_unitary
    #@+node:cog.20080728182049.21:compute_UL_for_isometry
    def compute_UL_for_isometry(self):
        I = identity(self.old_d)

        LLL_UL = self.make_UL_IC_left(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        LL_UL = self.make_UL_IC_right(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        L_UL = self.make_UL_IB_left(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        R_UL = self.make_UL_IB_right(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        RR_UL = self.make_UL_IA_left(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        RRR_UL = self.make_UL_IA_right(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        return LLL_UL+LL_UL+L_UL+R_UL+RR_UL+RRR_UL

    #@-node:cog.20080728182049.21:compute_UL_for_isometry
    #@+node:cog.20080728182049.22:fully_contract_term
    def fully_contract_term(self):
        I = identity(self.old_d)
        return self.fully_contract_left(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term
        ) + self.fully_contract_right(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term
        )
    #@-node:cog.20080728182049.22:fully_contract_term
    #@+node:gcross.20080730130632.6:compute_projector
    def compute_projector(self):
        I = identity(self.old_d)
        return self.make_left_projector(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.projector
        )+self.make_right_projector(
            I,I,I,I,
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.projector
        )
    #@-node:gcross.20080730130632.6:compute_projector
    #@+node:cog.20080728182049.23:renormalize_term
    def renormalize_term(self):
        return self.renormalize_left_H(
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term
        ) + self.renormalize_right_H(
            self.isometry,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term
        )
    #@-node:cog.20080728182049.23:renormalize_term
    #@-others
#@-node:cog.20080728182049.16:LayerG4
#@+node:cog.20080728182049.24:Layer4
class Layer4(Layer):
    #@    @+others
    #@+node:gcross.20080730142552.5:__init__
    def __init__(self,*args,**keywords):
        super(self.__class__,self).__init__(*args,**keywords)
        self.projector = identity(self.new_d**2).reshape((self.new_d,)*4)
    #@-node:gcross.20080730142552.5:__init__
    #@+node:cog.20080728182049.25:Contractors
    #@+others
    #@+node:cog.20080728182049.26:UL conversions
    g_left = make_placeholder_graph(
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(8,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IB*",(2,2,2),(13,14,15)),
        ("IC*",(2,2,2),(16,17,18)),
        ("UA*",(2,2,2,2),(18,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(31,-32,33,21,-22,23)),
        ("P",(2,2,2,2),(13,16,3,6)),
        ("UAI",(2,2),(22,-22)),
        ("UAI*",(2,2),(32,-32)),
        ("UBI",(2,2),(24,34)),
    )

    make_UL_IB_left = staticmethod(compile_graph(g_left,range(0,0) + range(1,4) + range(5,13),["UAI","UBI","IC","UA","UB","H","P"],node_ordering=[4,0]))
    make_UL_IC_left = staticmethod(compile_graph(g_left,range(0,1) + range(2,5) + range(6,13),["UAI","UBI","IB","UA","UB","H","P"],node_ordering=[5,1]))
    make_UL_UA_left = staticmethod(compile_graph(g_left,range(0,2) + range(3,6) + range(7,13),["UAI","UBI","IB","IC","UB","H","P"],node_ordering=[6,2]))
    make_UL_UB_left = staticmethod(compile_graph(g_left,range(0,3) + range(4,7) + range(8,13),["UAI","UBI","IB","IC","UA","H","P"],node_ordering=[7,3]))
    fully_contract_left = staticmethod(compile_graph(g_left,range(13),["UAI","UBI","IB","IC","UA","UB","H","P"],node_ordering=[]))
    make_left_projector = staticmethod(compile_graph(g_left,range(8)+range(9,13),["UAI","UBI","IB","IC","UA","UB","P"]))

    g_right = make_placeholder_graph(
        ("IB",(2,2,2),(3,4,5)),
        ("IC",(2,2,2),(6,7,8)),
        ("UA",(2,2,2,2),(8,4,21,22)),
        ("UB",(2,2,2,2),(5,7,23,24)),
        ("IB*",(2,2,2),(13,14,15)),
        ("IC*",(2,2,2),(16,17,18)),
        ("UA*",(2,2,2,2),(18,14,31,32)),
        ("UB*",(2,2,2,2),(15,17,33,34)),
        ("H",(2,2,2,2,2,2),(32,33,-34,22,23,-24)),
        ("P",(2,2,2,2),(13,16,3,6)),
        ("UBI",(2,2),(24,-24)),
        ("UBI*",(2,2),(34,-34)),
        ("UAI",(2,2),(21,31)),
    )

    make_UL_IB_right = staticmethod(compile_graph(g_right,range(0,0) + range(1,4) + range(5,13),["UAI","UBI","IC","UA","UB","H","P"],node_ordering=[4,0]))
    make_UL_IC_right = staticmethod(compile_graph(g_right,range(0,1) + range(2,5) + range(6,13),["UAI","UBI","IB","UA","UB","H","P"],node_ordering=[5,1]))
    make_UL_UA_right = staticmethod(compile_graph(g_right,range(0,2) + range(3,6) + range(7,13),["UAI","UBI","IB","IC","UB","H","P"],node_ordering=[6,2]))
    make_UL_UB_right = staticmethod(compile_graph(g_right,range(0,3) + range(4,7) + range(8,13),["UAI","UBI","IB","IC","UA","H","P"],node_ordering=[7,3]))
    fully_contract_right = staticmethod(compile_graph(g_right,range(13),["UAI","UBI","IB","IC","UA","UB","H","P"],node_ordering=[]))
    make_right_projector = staticmethod(compile_graph(g_right,range(8)+range(9,13),["UAI","UBI","IB","IC","UA","UB","P"]))
    #@-node:cog.20080728182049.26:UL conversions
    #@+node:cog.20080728182049.27:Renormalization tranformations
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
    #@-node:cog.20080728182049.27:Renormalization tranformations
    #@-others
    #@-node:cog.20080728182049.25:Contractors
    #@+node:cog.20080728182049.28:compute_UL_for_unitary
    def compute_UL_for_unitary(self):
        I = identity(self.old_d)

        next_left_UL = self.make_UL_UB_left(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        left_UL = self.make_UL_UB_right(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        right_UL = self.make_UL_UA_left(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        next_right_UL = self.make_UL_UA_right(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.H_term,
            self.projector
        )

        return next_left_UL+left_UL+right_UL+next_right_UL
    #@-node:cog.20080728182049.28:compute_UL_for_unitary
    #@+node:cog.20080728182049.29:compute_UL_for_isometry
    def compute_UL_for_isometry(self):
        I = identity(self.old_d)

        next_left_UL = self.make_UL_IC_left(
            I,I,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        left_UL = self.make_UL_IC_right(
            I,I,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        right_UL = self.make_UL_IB_left(
            I,I,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        next_right_UL = self.make_UL_IB_right(
            I,I,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )

        return next_left_UL+left_UL+right_UL+next_right_UL
    #@-node:cog.20080728182049.29:compute_UL_for_isometry
    #@+node:cog.20080728182049.30:fully_contract_term
    def fully_contract_term(self):
        I = identity(self.old_d)

        return self.fully_contract_left(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        ) + self.fully_contract_right(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term,
            self.projector
        )
    #@-node:cog.20080728182049.30:fully_contract_term
    #@+node:gcross.20080730130632.5:compute_projector
    def compute_projector(self):
        I = identity(self.old_d)

        return self.make_left_projector(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.projector
        )+self.make_right_projector(
            I,I,
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.projector
        )

    #@-node:gcross.20080730130632.5:compute_projector
    #@+node:cog.20080728182049.31:renormalize_term
    def renormalize_term(self):
        return self.renormalize_left_H(
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term
        ) + self.renormalize_right_H(
            self.isometry,
            self.isometry,
            self.unitary,
            self.unitary,
            self.H_term
        )
    #@-node:cog.20080728182049.31:renormalize_term
    #@-others
#@-node:cog.20080728182049.24:Layer4
#@+node:cog.20080728182049.32:TopLayer
class TopLayer(object):
    #@    @+others
    #@+node:cog.20080728182049.33:__init__
    def __init__(self,H_term,new_d,parent):
        self.parent = parent
        self.set_H_term(H_term)
        self.new_d = new_d

    #@-node:cog.20080728182049.33:__init__
    #@+node:gcross.20080730142552.2:set_H_term
    def set_H_term(self,H_term):
        H = H_term + H_term.transpose(1,0,3,2)
        H = H.reshape(H.shape[0]*H.shape[1],H.shape[2]*H.shape[3])
        self.H = H
        self.H_term = H_term
    #@-node:gcross.20080730142552.2:set_H_term
    #@+node:cog.20080728182049.34:trace
    def trace(self):
        return sum(self.spectrum())



    #@-node:cog.20080728182049.34:trace
    #@+node:cog.20080728182049.35:spectrum
    def spectrum(self):
        evals = eigvalsh(self.H)
        return evals[argsort(evals)[-self.new_d:]]
    #@-node:cog.20080728182049.35:spectrum
    #@+node:gcross.20080730142552.6:project
    def project(self):
        evals, evecs = eigh(self.H)
        indices = argsort(evals)[-self.new_d:]
        evecs_of_interest = evecs[:,indices]
        self.parent.projector = dot(evecs_of_interest,evecs_of_interest.conj().transpose()).reshape(self.H_term.shape)
    #@-node:gcross.20080730142552.6:project
    #@-others
#@-node:cog.20080728182049.32:TopLayer
#@-others
#@-node:cog.20080728182049.5:Classes
#@+node:cog.20080728182049.36:Initialization
H_term = (reduce(multiply.outer,[0.5*X,X,I])+reduce(multiply.outer,[Z,I,I])).transpose(0,2,4,1,3,5)

#new_d = 3

#basement_layer = LayerG4(H_term,16,new_d)
#bottom_layer = basement_layer.create_renormalized_layer(new_d)
bottom_layer = LayerG4(H_term,8,3)
middle_layer = bottom_layer.create_renormalized_layer(6)
top_layer = middle_layer.create_renormalized_layer(4)

#@nonl
#@-node:cog.20080728182049.36:Initialization
#@+node:cog.20080728182049.37:Build MERA
#basement_layer.optimize(10)
#basement_layer.renormalize()
bottom_layer.optimize(10)
bottom_layer.renormalize()
middle_layer.optimize(10)
middle_layer.renormalize()
print top_layer.trace(), top_layer.spectrum()

print "*************"
print "MERA BUILT!!!"
print "*************"
#@-node:cog.20080728182049.37:Build MERA
#@+node:gcross.20080730142552.4:Sweep MERA
for dummy in xrange(10):
    top_layer.project()
    middle_layer.optimize(10)
    middle_layer.project()
    bottom_layer.optimize(10)
    #bottom_layer.project()
    #basement_layer.optimize(10)
    #basement_layer.renormalize()
    bottom_layer.optimize(10)
    bottom_layer.renormalize()
    middle_layer.optimize(10)
    middle_layer.renormalize()

    print top_layer.trace(), top_layer.spectrum()

#@-node:gcross.20080730142552.4:Sweep MERA
#@-others
#@-node:cog.20080728182049.3:@thin MERA_1d_Ising_TI.py
#@-leo
