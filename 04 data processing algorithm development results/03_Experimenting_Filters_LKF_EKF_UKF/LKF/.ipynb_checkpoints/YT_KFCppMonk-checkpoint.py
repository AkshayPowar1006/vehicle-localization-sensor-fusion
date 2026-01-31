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
   "execution_count": 2,
   "id": "310c05b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import unittest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
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
    "        print(new_x)\n",
    "        print(new_P)\n",
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
   "execution_count": 38,
   "id": "5cf2ebc6",
   "metadata": {},
   "outputs": [],
   "source": [
    "kf = KF(init_x = 20, init_v=0.5, acc_noise = 0.05)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "7f7dacf1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[20.25  0.5 ]\n",
      "[[1.25078125 0.503125  ]\n",
      " [0.503125   1.0125    ]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "kf.predict(0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea8c6456",
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
