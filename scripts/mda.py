# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm, gamma
import random
import sys
import copy

from rand_utils import rand_partition_log
from hmc import hmc

class MatrixDecompositionAutologistic(object):
    S_X = 1
    S_Z = 2
    S_W_MH = 3
    S_W_HMC = 4
    S_Z_V = 5
    S_Z_H = 6
    S_Z_A = 7

    HMC_L = 10
    HMC_EPSILON = 0.05 # 0.1
    # BETA_A = 1.0
    # BETA_B = 1.0

    def __init__(self, mat, M, fmap, bmap,
                 sigma=1.0,
                 vnet=None, hnet=None,
                 # beta_a = self.BETA_A,
                 # beta_b = self.BETA_B,
                 norm_sigma = 5.0,
                 gamma_shape = 1.0,
                 gamma_scale = 0.001,
                 K=50, mvs=None,
                 only_alphas=False,
                 drop_vs=False,
                 drop_hs=False,
    ):
        self.mat = mat # X: L x N matrix
        self.vnet = vnet
        self.hnet = hnet
        self.only_alphas = only_alphas
        self.drop_vs = drop_vs
        self.drop_hs = drop_hs
        self.M = M
        (self.L, self.P) = self.mat.shape
        assert(mvs is None or mat.shape == mvs.shape)
        self.mvs = mvs # missing values; (i,p) => bool (True: missing value)
        self.mv_list = []
        if self.mvs is not None:
            for l in xrange(self.L):
                for p in xrange(self.P):
                    if self.mvs[l,p]:
                        self.mv_list.append((l,p))
        self.fmap = fmap # fmap(j) = p, q
        self.bmap = bmap # bmap(p) = j_start, T
        self.K = K
        # self.beta_a = beta_a
        # self.beta_b = beta_b
        # np.random.beta(self.beta_a, self.beta_b, size=self.K)
        self.norm_sigma = norm_sigma
        self.alphas = 0.5 * np.random.normal(loc=0.0, scale=self.norm_sigma, size=self.K)
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        if not (self.only_alphas or self.drop_vs):
            self.vks = 0.0001 * np.ones(self.K, dtype=np.float32)
        else:
            self.vks = np.zeros(self.K, dtype=np.float32)
        if not (self.only_alphas or self.drop_hs):
            self.hks = 0.0001 * np.ones(self.K, dtype=np.float32)
        else:
            self.hks = np.zeros(self.K, dtype=np.float32)
        self.zmat = np.zeros((self.K, self.L), dtype=np.bool_)
        for k, alpha in enumerate(self.alphas):
            thres = 1.0 / (1.0 + np.exp(-alpha))
            self.zmat[k] = (np.random.rand(self.L) < thres)
        self.sigma = sigma # Normal
        self.wmat = 0.1 * np.random.standard_t(df=self.sigma, size=(self.K, self.M)) # np.random.normal(loc=0.0, scale=0.1, size=(self.K, self.M))
        self.theta_tilde = np.zeros((self.L, self.M), dtype=np.float32)
        self.theta = np.ones((self.L, self.M), dtype=np.float32)
        self.calc_theta_tilde()
        self.init_tasks()

    def dumps(self):
        import copy
        obj = copy.copy(self)
        del obj.fmap
        del obj.bmap
        return obj

    def init_dump(self, fmap, bmap):
        self.fmap = fmap
        self.bmap = bmap

    def init_with_freq(self, K=50, anneal=0.0):
        # arg: K is dummy
        freqlist = 0.5 * np.ones(self.M, dtype=np.float32)
        for i in xrange(self.L):
            for p in xrange(self.P):
                j_start, T = self.bmap(p)
                if self.mvs is None or self.mvs[i,p] == False:
                    freqlist[j_start+self.mat[i,p]] += 1
        for p in xrange(self.P):
            j_start, T = self.bmap(p)
            freqlist[j_start:j_start+T] /= freqlist[j_start:j_start+T].sum()
            if anneal > 0.0:
                freqlist[j_start:j_start+T] = freqlist[j_start:j_start+T] ** anneal
                freqlist[j_start:j_start+T] /= freqlist[j_start:j_start+T].sum()
        logfreqlist = np.log(freqlist)
        alpha = self.K / 10.0 # magic
        mu = 0.99
        idxs = np.random.randint(0, self.L, size=K)
        betas = np.random.beta(alpha, 1.0, size=self.L)
        ws = np.zeros((self.M, K), dtype=np.float32)
        for j in xrange(self.M):
            ws[j] = np.random.normal(logfreqlist[j], scale=0.5, size=K) # mean w_old, variance: self.GAMMA_ETA * w_old
        ws = np.where(ws > 1E-3, ws, 1E-3)
        for k, (idx, beta) in enumerate(zip(idxs, betas)):
            # this decays too fast
            if min_mu < 1E-3:
                min_mu *= 0.99
            else:
                min_mu *= beta
            sys.stderr.write("{}\n".format(min_mu))
            self.alphas[k] = np.log(min_mu / (1.0 - min_mu))
            self.zmat[k] = (np.random.random_sample(self.L) < min_mu)
            sys.stderr.write("{}\n".format(zmat[k].sum()))
            self.wmat[k] = np.random.standard_t(df=self.sigma, size=self.M)
            self.active_features[id(feature)] = feature
        self.calc_theta_tilde()
        self.sample_mus()

    def init_with_clusters(self, K=50):
        # arg: K is dummy
        freqlist = np.zeros(self.M, dtype=np.float32)
        for i in xrange(self.L):
            for p in xrange(self.P):
                j_start, T = self.bmap(p)
                if self.mvs is None or self.mvs[i,p] == False:
                    freqlist[j_start+self.mat[i,p]] += 1
        for p in xrange(self.P):
            j_start, T = self.bmap(p)
            freqlist[j_start:j_start+T] /= freqlist[j_start:j_start+T].sum()
        # use only K-1 binary features
        killlist = np.arange(self.M)
        np.random.shuffle(killlist)
        for i in xrange(self.M - K + 1):
            freqlist[killlist[i]] = 0.0
        jlist = sorted(range(self.M), key=lambda x: freqlist[x], reverse=True)
        min_mu = 0.99
        # 1st feature: fully active
        self.alphas[0] = np.log(min_mu / (1.0 - min_mu))
        self.zmat[0] = True
        self.wmat[0] = np.random.normal(loc=0.0, scale=0.1, size=self.M)

        # subsequent K-1 features
        idxs = np.random.randint(0, self.L, size=K)
        for k in xrange(1, K):
            j = jlist[k-1]
            min_mu = max(freqlist[j], 0.001)
            self.alphas[k] = np.log(min_mu / (1.0 - min_mu))
            self.wmat[k] = np.random.normal(loc=0.0, scale=0.1, size=self.M)
            self.wmat[k,j] += 10.0 * np.random.gamma(freqlist[j] * 1.0, 1.0)
            p, q = self.fmap(j)
            j_start, T = self.bmap(p)
            for l in xrange(self.L):
                if self.mvs is None or self.mvs[l,p] == False:
                    if self.mat[l,p] == q:
                        self.zmat[k,l] = True
                    else:
                        self.zmat[k,l] = False
                else:
                    self.zmat[k,l] = (np.random.rand() < min_mu)
            # feature.Lvect[idxs[k]] = True # used by at least one language
            sys.stderr.write("{}\n".format(min_mu))
            sys.stderr.write("{}\n".format(self.zmat[k].sum()))
        self.calc_theta_tilde()

    def calc_loglikelihood(self):
        # self.calc_theta_tilde()
        ll = 0.0
        for i in xrange(self.L):
            for p in xrange(self.P):
                j_start, T = self.bmap(p)
                x = self.mat[i,p]
                ll += np.log(self.theta[i,j_start+x] + 1E-20)
                # theta_tilde2 = self.theta_tilde[i,j_start:j_start+T] - self.theta_tilde[i,j_start:j_start+T].max()
                # e_theta_tilde = np.exp(theta_tilde2)
                # ll += theta_tilde2[x] - np.log(e_theta_tilde.sum())
        return ll

    def calc_theta_tilde(self):
        self.theta_tilde[...] = np.matmul(self.zmat.T, self.wmat) # (K x L)^T x (K x M) -> (L x M)
        for p in xrange(self.P):
            j_start, T = self.bmap(p)
            e_theta_tilde = np.exp(self.theta_tilde[:,j_start:j_start+T] - self.theta_tilde[:,j_start:j_start+T].max(axis=1).reshape(self.L, 1))
            self.theta[:,j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum(axis=1).reshape(self.L, 1)
        # for i in xrange(self.L):
        #     for p in xrange(self.P):
        #         j_start, T = self.bmap(p)
        #         e_theta_tilde = np.exp(self.theta_tilde[i,j_start:j_start+T] - self.theta_tilde[i,j_start:j_start+T].max())
        #         self.theta[i,j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum()

    def init_tasks(self, a_repeat=1, sample_w=True):
        self.tasks = map(lambda x: (self.S_X, x), self.mv_list)
        for k in xrange(self.K):
            for a in xrange(a_repeat):
                if not self.only_alphas:
                    if not self.drop_vs:
                        self.tasks.append((self.S_Z_V, k))
                    if not self.drop_hs:
                        self.tasks.append((self.S_Z_H, k))
                self.tasks.append((self.S_Z_A, k))
        for l in xrange(self.L):
            self.tasks += map(lambda k: (self.S_Z, (l, k)), xrange(self.K))
        if sample_w:
            self.tasks += map(lambda k: (self.S_W_HMC, k), xrange(self.K))

    def sample(self, _iter=0, maxanneal=0, itemp=-1):
        # inverse of temperature
        if itemp > 0:
            sys.stderr.write("\t\titemp\t{}\n".format(itemp))
        elif _iter >= maxanneal:
            itemp = 1.0
        else:
            itemp = 0.1 + 0.9 * _iter / maxanneal
            sys.stderr.write("\t\titemp\t{}\n".format(itemp))

        c_x = [0, 0]
        c_z = [0, 0]
        c_z_v = [0, 0]
        c_z_h = [0, 0]
        c_z_a = [0, 0]
        c_w_hmc = [0, 0]
        random.shuffle(self.tasks)
        for t_type, t_val in self.tasks:
            if t_type == self.S_X:
                l, p = t_val
                changed = self.sample_x(l, p)
                c_x[changed] += 1
            elif t_type == self.S_Z:
                l, k = t_val
                changed = self.sample_z(l, k, itemp=itemp)
                c_z[changed] += 1
            elif t_type == self.S_W_HMC:
                changed = self.sample_w_hmc(t_val)
                c_w_hmc[changed] += 1
            elif t_type == self.S_Z_V:
                changed = self.sample_autologistic(t_type, t_val)
                c_z_v[changed] += 1
            elif t_type == self.S_Z_H:
                changed = self.sample_autologistic(t_type, t_val)
                c_z_h[changed] += 1
            elif t_type == self.S_Z_A:
                changed = self.sample_autologistic(t_type, t_val)
                c_z_a[changed] += 1
            else:
                raise NotImplementedError
        self.calc_theta_tilde() # fix numerical errors
        sys.stderr.write("\tx\t%f\n" % (float(c_x[1]) / sum(c_x)))
        sys.stderr.write("\tz\t%f\n" % (float(c_z[1]) / sum(c_z)))
        if sum(c_w_hmc) > 0:
            sys.stderr.write("\tw_hmc\t%f\n" % (float(c_w_hmc[1]) / sum(c_w_hmc)))
        if not self.only_alphas:
            if sum(c_z_v) > 0:
                sys.stderr.write("\tz_v\t%f\n" % (float(c_z_v[1]) / sum(c_z_v)))
            if sum(c_z_h) > 0:
                sys.stderr.write("\tz_h\t%f\n" % (float(c_z_h[1]) / sum(c_z_h)))
        if sum(c_z_a) > 0:
            sys.stderr.write("\tz_a\t%f\n" % (float(c_z_a[1]) / sum(c_z_a)))
        if not self.only_alphas:
            sys.stderr.write("\tv\tavg\t%f\tmax\t%f\n" % (self.vks.mean(), self.vks.max()))
            sys.stderr.write("\th\tavg\t%f\tmax\t%f\n" % (self.hks.mean(), self.hks.max()))
        sys.stderr.write("\ta\tavg\t%f\tvar\t%f\n" % (self.alphas.mean(), self.alphas.var()))

    def sample_x(self, i, p):
        assert(self.mvs is not None and self.mvs[i,p])
        j_start, T = self.bmap(p)
        x_old = self.mat[i,p]
        self.mat[i,p] = np.random.choice(T, p=self.theta[i,j_start:j_start+T])
        return False if x_old == self.mat[i,p] else True

    def sample_z(self, l, k, itemp=1.0, frozen=False):
        z_old = self.zmat[k,l]
        logprob0, logprob1 = (0.0, 0.0)
        if not self.only_alphas:
            varray = self.zmat[k,self.vnet[l]]
            logprob0 += self.vks[k] * (varray == False).sum()
            logprob1 += self.vks[k] * (varray == True).sum()
            harray = self.zmat[k,self.hnet[l]]
            logprob0 += self.hks[k] * (harray == False).sum()
            logprob1 += self.hks[k] * (harray == True).sum()
        logprob1 += self.alphas[k]
        for p in xrange(self.P):
            j_start, T = self.bmap(p)
            x = self.mat[l,p]
            prob_x = self.theta[l,j_start+x] + 1E-20
            if z_old == False:
                logprob0 += np.log(prob_x)
                theta_tilde2 = self.theta_tilde[l,j_start:j_start+T] + self.wmat[k,j_start:j_start+T]
                e_theta_tilde2 = np.exp(theta_tilde2 - theta_tilde2.max())
                prob_x2 = (e_theta_tilde2 / e_theta_tilde2.sum())[x] + 1E-20
                logprob1 += np.log(prob_x2)
            else:
                logprob1 += np.log(prob_x)
                theta_tilde2 = self.theta_tilde[l,j_start:j_start+T] - self.wmat[k,j_start:j_start+T]
                e_theta_tilde2 = np.exp(theta_tilde2 - theta_tilde2.max())
                # assert(np.all(theta_tilde2 > 0))
                prob_x2 = (e_theta_tilde2 / e_theta_tilde2.sum())[x] + 1E-20
                logprob0 += np.log(prob_x2)
        if itemp != 1.0:
            logprob0 *= itemp
            logprob1 *= itemp
        z_new = np.bool_(rand_partition_log((logprob0, logprob1)))
        self.zmat[k,l] = z_new
        if z_old == z_new:
            return False
        else:
            if z_new == True:
                # 0 -> 1
                self.theta_tilde[l] += self.wmat[k]
            else:
                # 1 -> 0
                self.theta_tilde[l] -= self.wmat[k]
            for p in xrange(self.P):
                j_start, T = self.bmap(p)
                e_theta_tilde = np.exp(self.theta_tilde[l,j_start:j_start+T] - self.theta_tilde[l,j_start:j_start+T].max())
                self.theta[l,j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum()
            return True

    def sample_autologistic(self, t_type, k):
        logr = 0.0
        if t_type == self.S_Z_A:
            oldval = self.alphas[k]
            pivot = min((self.zmat[k].sum() + 0.01) / self.L, 0.99)
            pivot = np.log(pivot / (1.0 - pivot))
            oldmean = (oldval + pivot) / 2.0
            oldscale = max(abs(oldval - pivot), 0.001)
            newval = np.random.normal(loc=oldmean, scale=oldscale)
            newmean = (newval + pivot) / 2.0
            newscale = max(abs(newval - pivot), 0.001)
            # q(theta|theta', x) / q(theta'|theta, x)
            logr += -((oldval - newmean) ** 2) / (2.0 * newscale * newscale) - np.log(newscale) \
                    + ((newval - oldmean) ** 2) / (2.0 * oldscale * oldscale) + np.log(oldscale)
            # P(theta') / P(theta)
            logr += (oldval * oldval - newval * newval) / (2.0 * self.norm_sigma * self.norm_sigma)
            # skip: q(theta|theta', x) / q(theta'|theta, x) for symmetric proposal
            v, h, a = self.vks[k], self.hks[k], newval
        else:
            assert(not self.only_alphas)
            assert(not (t_type == self.S_Z_V and self.drop_vs))
            assert(not (t_type == self.S_Z_H and self.drop_hs))
            if t_type == self.S_Z_V:
                oldval = self.vks[k]
            else:
                oldval = self.hks[k]
            P_SIGMA = 0.5
            rate = np.random.lognormal(mean=0.0, sigma=P_SIGMA)
            irate = 1.0 / rate
            newval = rate * oldval
            lograte = np.log(rate)
            logirate = np.log(irate)
            # P(theta') / P(theta)
            # logr += gamma.logpdf(newval, self.gamma_shape, scale=self.gamma_scale) \
            #         - gamma.logpdf(oldval, self.gamma_shape, scale=self.gamma_scale)
            logr += (self.gamma_shape - 1.0) * (np.log(newval) - np.log(oldval)) \
                    - (newval - oldval) / self.gamma_scale
            # q(theta|theta', x) / q(theta'|theta, x)
            logr += (lograte * lograte - logirate * logirate) / (2.0 * P_SIGMA * P_SIGMA) + lograte - logirate
            if t_type == self.S_Z_V:
                v, h, a = newval, self.hks[k], self.alphas[k]
                net = self.vnet
            else:
                v, h, a = self.vks[k], newval, self.alphas[k]
                net = self.hnet
        zvect = self.zmat[k].copy()
        llist = np.arange(self.L)
        np.random.shuffle(llist)
        for l in llist:
            logprob0, logprob1 = (0.0, 0.0)
            if not self.only_alphas:
                varray = zvect[self.vnet[l]]
                logprob0 += v * (varray == False).sum()
                logprob1 += v * (varray == True).sum()
                harray = zvect[self.hnet[l]]
                logprob0 += h * (harray == False).sum()
                logprob1 += h * (harray == True).sum()
            logprob1 += a
            zvect[l] = rand_partition_log([logprob0, logprob1])
        # V_oldsum = self._neighbor_sum(self.zmat[k], self.vnet)
        # H_oldsum = self._neighbor_sum(self.zmat[k], self.hnet)
        # A_oldsum = self.zmat[k].sum()
        # V_newsum = self._neighbor_sum(zvect, self.vnet)
        # H_newsum = self._neighbor_sum(zvect, self.hnet)
        # A_newsum = zvect.sum()
        # logr += (self.vks[k] * V_newsum + self.hks[k] * H_newsum + self.alphas[k] * A_newsum) \
        #         + (v * V_oldsum + h * H_oldsum + a * A_oldsum) \
        #         - (self.vks[k] * V_oldsum + self.hks[k] * H_oldsum + self.alphas[k] * A_oldsum) \
        #         - (v * V_newsum + h * H_newsum + a * A_newsum)
        # logr += (self.vks[k] * (V_newsum - V_oldsum) + self.hks[k] * (H_newsum - H_oldsum) + self.alphas[k] * (A_newsum - A_oldsum)) \
        #         - (v * (V_newsum - V_oldsum) + h * (H_newsum - H_oldsum) + a * (A_newsum - A_oldsum))
        # logr += ((self.vks[k] - v) * (V_newsum - V_oldsum) + (self.hks[k] - h) * (H_newsum - H_oldsum) + (self.alphas[k] - a) * (A_newsum - A_oldsum))
        if t_type == self.S_Z_A:
            logr += (oldval - newval) * (zvect.sum() - self.zmat[k].sum())
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                self.alphas[k] = newval
                return True
            else:
                return False
        else:
            oldsum = self._neighbor_sum(self.zmat[k], net)
            newsum = self._neighbor_sum(zvect, net)
            logr += (oldval - newval) * (newsum - oldsum)
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                if t_type == self.S_Z_V:
                    self.vks[k] = newval
                else:
                    self.hks[k] = newval
                return True
            else:
                return False

    def _neighbor_sum(self, zvect, net):
        s = 0
        for l in xrange(self.L):
            s += (zvect[net[l]] == zvect[l]).sum()
        assert(s % 2 == 0)
        return s / 2

    def sample_w_hmc(self, k):
        def U(Mvect):
            # ll = -norm.logpdf(Mvect, 0.0, scale=self.sigma).sum()
            ll = 0.5 * (self.sigma + 1.0) * np.log(1.0 + (Mvect * Mvect) / self.sigma).sum()
            for l in xrange(self.L):
                if self.zmat[k,l] == False:
                    continue
                theta_tilde = self.theta_tilde[l] - self.wmat[k] + Mvect
                for p in xrange(self.P):
                    j_start, T = self.bmap(p)
                    x = self.mat[l,p]
                    theta_tilde2 = theta_tilde[j_start:j_start+T] - theta_tilde[j_start:j_start+T].max()
                    ll -= theta_tilde2[x] - np.log(np.exp(theta_tilde2).sum())
            return ll
        sigma2 = (self.sigma + 1.0) / self.sigma
        def gradU(Mvect):
            grad = sigma2 * (Mvect / (1.0 + (Mvect * Mvect) / self.sigma))
            for l in xrange(self.L):
                if self.zmat[k,l] == False:
                    continue
                theta_tilde = self.theta_tilde[l] - self.wmat[k] + Mvect
                for p in xrange(self.P):
                    j_start, T = self.bmap(p)
                    x = self.mat[l,p]
                    j = j_start + x
                    e_theta_tilde = np.exp(theta_tilde[j_start:j_start+T] - theta_tilde[j_start:j_start+T].max())
                    theta = e_theta_tilde / e_theta_tilde.sum()
                    grad[j_start:j_start+T] += theta
                    grad[j] -= 1
            return grad
        accepted, Mvect = hmc(U, gradU, self.HMC_EPSILON, self.HMC_L, self.wmat[k])
        if accepted:
            # update theta_tilde
            for l in xrange(self.L):
                if self.zmat[k,l] == False:
                    continue
                self.theta_tilde[l] += Mvect - self.wmat[k]
                for p in xrange(self.P):
                    j_start, T = self.bmap(p)
                    e_theta_tilde = np.exp(self.theta_tilde[l,j_start:j_start+T] - self.theta_tilde[l,j_start:j_start+T].max())
                    self.theta[l,j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum()
                # assert(all(self.theta_tilde[i] > 0))
            self.wmat[k] = Mvect
            return True
        else:
            return False
