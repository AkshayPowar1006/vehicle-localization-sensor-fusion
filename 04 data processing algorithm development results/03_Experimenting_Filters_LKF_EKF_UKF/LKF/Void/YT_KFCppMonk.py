{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a7731f86",
   "metadata": {},
   "source": [
    "1. Design KF\n",
    "Goal: Estimate X and V\n",
    "\n",
    "measurements Z = X + epsilon (GaussiNoise, mu = 0 and sigma^2 is variance)\n",
    "KF not only gives X and V but also uncertainity of the object\n",
    "\n",
    "Xk is staet vector, Zk is measurement vector \n",
    "\n",
    "1: Time Evaluation\n",
    "u is always the noisy input to the system which casues the acceleration into the system this is normal distribution with mean as zero and some variance sigma square\n",
    "\n",
    "2: Measurement Update:\n",
    "Zk updating\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "310c05b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import unittest\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "d4573937",
   "metadata": {},
   "outputs": [],
   "source": [
    "class KF:\n",
    "    def __init__(self, init_x: float, \n",
    "                 init_v: float,\n",
    "                acc_noise : float):\n",
    "        \n",
    "        # Mean of state GRV\n",
    "        self.x = np.array([init_x, init_v])\n",
    "        self.acc_noise = acc_noise\n",
    "        \n",
    "        # Covariance of state GRV\n",
    "        self.P = np.eye(2)\n",
    "        \n",
    "    def predict(self, dt:float) -> None:\n",
    "        # x = F * x\n",
    "        # P = F * P * Ft + G * a * Gt\n",
    "        F = np.array([[1, dt], [0,1]])\n",
    "        G = np.array([[dt**2/2],[dt]])\n",
    "        new_x = np.dot(F,self.x)\n",
    "        new_P = F.dot(self.P).dot(F.T) + G.dot(self.acc_noise).dot(G.T)\n",
    "        \n",
    "        self.x = new_x\n",
    "        self.P = new_P\n",
    "        \n",
    "        #rint(new_x)\n",
    "        #rint(new_P)\n",
    "        \n",
    "    def update(self, meas_value: float, meas_variance: float):\n",
    "        # y = z- Hx\n",
    "        # S= H P Ht + R\n",
    "        # K= P Ht S^2-1\n",
    "        # x = x = Ky\n",
    "        # P = (I -KH) P\n",
    "        H = np.array([1,0]).reshape(1,2)\n",
    "        z = np.array([meas_value])\n",
    "        R = np.array([meas_variance])\n",
    "        \n",
    "        y = z - H.dot(self.x)\n",
    "        S = H.dot(self.P).dot(H.T) + R \n",
    "        \n",
    "        K = self.P.dot(H.T).dot(np.linalg.inv(S))\n",
    "        \n",
    "        new_x = self.x + K.dot(y)\n",
    "        new_P = ((np.eye(2)- K.dot(H))).dot(self.x)\n",
    "         \n",
    "        self.x = new_x\n",
    "        self.P = new_P\n",
    "        print(meas_value)\n",
    "        \n",
    "       \n",
    "        \n",
    "        \n",
    "    \n",
    "    @property\n",
    "    def cov(self) -> np.array:\n",
    "        return self.P\n",
    "    \n",
    "    @property\n",
    "    def mean(self) -> np.array:\n",
    "        return self.x\n",
    "        \n",
    "    @property\n",
    "    def pos(self) -> float:\n",
    "        return self.x[0]\n",
    "    @property\n",
    "    def vel(self) -> float:\n",
    "        return self.x[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d81c702",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2082164",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f6d34cd0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "24.03070104314215\n",
      "27.903387057633296\n",
      "31.53142176180886\n",
      "35.537519006767134\n",
      "39.19744996185626\n",
      "43.10764424676736\n",
      "46.93735787355117\n",
      "50.643906376520114\n",
      "54.249337197844035\n",
      "58.06460630782748\n",
      "62.07670830008212\n",
      "65.62780117933659\n",
      "69.52495304324873\n",
      "73.18481468270532\n",
      "77.08098653887868\n",
      "80.80760345397877\n",
      "84.66462751592385\n",
      "88.56962150371051\n",
      "92.19903315015661\n",
      "96.04048463788322\n",
      "100.1674032049106\n",
      "103.57775107132078\n",
      "107.69833986734295\n",
      "111.34779653358017\n",
      "115.24551627814496\n",
      "118.81755001259829\n",
      "122.6827856928572\n",
      "126.53417161507633\n",
      "130.2656723490242\n",
      "134.17717931833536\n",
      "137.90526226156075\n",
      "141.94205532880343\n",
      "145.64568388276408\n",
      "149.3734972801581\n",
      "153.06650424844193\n",
      "156.97012359490307\n",
      "160.8690361462302\n",
      "164.7916321236447\n",
      "168.2847212760503\n",
      "172.42916275251474\n",
      "176.04314703376937\n",
      "179.66870687843502\n",
      "183.74048065568385\n",
      "187.34788954166524\n",
      "191.17514710133207\n",
      "194.6772642488233\n",
      "198.79409137211502\n",
      "202.7958955492187\n",
      "206.36972037828582\n"
     ]
    },
    {
     "ename": "IndexError",
     "evalue": "too many indices for array: array is 1-dimensional, but 2 were indexed",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_14340\\2003300714.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     33\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Position\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     34\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mmu\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mmu\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mmus\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'r'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 35\u001b[1;33m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mmu\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m-\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msqrt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcov\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mmu\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcov\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmus\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcovs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'r--'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     36\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mmu\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m+\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msqrt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcov\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mmu\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcov\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmus\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcovs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'r--'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     37\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_14340\\2003300714.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m     33\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Position\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     34\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mmu\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mmu\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mmus\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'r'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 35\u001b[1;33m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mmu\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m-\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msqrt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcov\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mmu\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcov\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmus\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcovs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'r--'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     36\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mmu\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m+\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msqrt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcov\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mmu\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcov\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmus\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcovs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'r--'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     37\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mIndexError\u001b[0m: too many indices for array: array is 1-dimensional, but 2 were indexed"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.ion()\n",
    "plt.figure()\n",
    "\n",
    "kf = KF(init_x= 20.0, init_v = 2, acc_noise = 0.1)\n",
    "\n",
    "time_steps = 0.1\n",
    "num_steps = 1000\n",
    "meas_steps = 20\n",
    "\n",
    "real_x = 20.0\n",
    "meas_variance = 0.02\n",
    "real_v = 1.9\n",
    "\n",
    "mus = []\n",
    "covs = []\n",
    "\n",
    "for step in range(num_steps):\n",
    "    covs.append(kf.cov)\n",
    "    mus.append(kf.mean)\n",
    "    \n",
    "    real_x = real_x + time_steps * real_v\n",
    "    #print(real_x)\n",
    "    \n",
    "    kf.predict(dt = time_steps)\n",
    "    if step!= 0 and step % meas_steps == 0:\n",
    "        kf.update(meas_value=real_x + np.random.randn()*np.sqrt(meas_variance),\n",
    "                  meas_variance= meas_variance)\n",
    "        \n",
    "        \n",
    "    \n",
    "  \n",
    "plt.subplot(2,1,1)\n",
    "plt.title(\"Position\")\n",
    "plt.plot([mu[0] for mu in mus], 'r')\n",
    "plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu,cov in zip(mus, covs)], 'r--')\n",
    "plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu,cov in zip(mus, covs)], 'r--')\n",
    "\n",
    "\n",
    "plt.subplot(2,1,2)\n",
    "plt.title(\"Velocity\")\n",
    "plt.plot([mu2[1] for mu2 in mus], 'g')\n",
    "plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu,cov in zip(mus, covs)], 'g--')\n",
    "plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu,cov in zip(mus, covs)], 'g--')\n",
    "plt.show()\n",
    "\n",
    "#plt.ginput(1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "92dac6dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "kf = KF(init_x= 20.0, init_v = 2, acc_noise = 0.1)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e678abb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1],\n",
       "       [0]])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "H"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "603136d7",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "shapes (2,1) and (2,) not aligned: 1 (dim 1) != 2 (dim 0)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_14340\\585664529.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m kf.update(meas_value=real_x + np.random.randn()*np.sqrt(meas_variance),\n\u001b[0m\u001b[0;32m      2\u001b[0m                   meas_variance= meas_variance)\n",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_14340\\3975296152.py\u001b[0m in \u001b[0;36mupdate\u001b[1;34m(self, meas_value, meas_variance)\u001b[0m\n\u001b[0;32m     35\u001b[0m         \u001b[0mR\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mmeas_variance\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     36\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 37\u001b[1;33m         \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mz\u001b[0m \u001b[1;33m-\u001b[0m \u001b[0mH\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     38\u001b[0m         \u001b[0mS\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mH\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mP\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mH\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mT\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mR\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     39\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: shapes (2,1) and (2,) not aligned: 1 (dim 1) != 2 (dim 0)"
     ]
    }
   ],
   "source": [
    "kf.update(meas_value=real_x + np.random.randn()*np.sqrt(meas_variance),\n",
    "                  meas_variance= meas_variance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "7de10f45",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'mu' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_14340\\81818669.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmu\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'mu' is not defined"
     ]
    }
   ],
   "source": [
    "mu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4be5237a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
