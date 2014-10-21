#########################
#########################
# load data
import numpy as np
from sklearn.decomposition import PCA

a = np.loadtxt('./ledaoutput.txt', dtype='string', delimiter='|', skiprows=3)
good  = np.arange(len(a[:,0]))[(a[:,1] != '                 ') &  (a[:,2] != '                 ') & (a[:,3] != '                 ')]

vdisp = a[good,1].astype('float')
radius = a[good,2].astype('float')
absm = a[good,3].astype('float')

#rescale, center at 0
vdispr  = vdisp-np.mean(vdisp)
vdispr  = vdispr/np.var(vdispr)
radiusr = radius-np.mean(radius)
radiusr = radiusr/np.var(radiusr)
absmr   = absm-np.mean(absm)
absmr   = absmr/np.var(absmr)

#compute PCA components
matrix = np.vstack([vdispr, radiusr, absmr])
pca = PCA(n_components=3)
pca.fit(np.transpose(matrix))
comp = pca.components_

new1 = np.dot(np.transpose(matrix), comp[0])
new2 = np.dot(np.transpose(matrix), comp[1])
new3 = np.dot(np.transpose(matrix), comp[2])


###########################
####plot
plt.figure()
plt.subplot(2,2,1)
plt.scatter(vdisp, radius, color = 'k', alpha = 0.6)
#plt.xlabel('Velocity dispersion [km/s]')
plt.ylabel('Apparent radius [degrees]')
plt.ylim([21,27])
plt.locator_params(nbins=4)
plt.subplot(2,2,3)
plt.scatter(vdisp, absm, color = 'k', alpha = 0.6)
plt.xlabel('Velocity dispersion [km/s]')
plt.ylabel('Absolute magnitude [mag]')
plt.locator_params(nbins=4)
plt.subplot(2,2,4)
plt.scatter(radius,absm, color = 'k', alpha = 0.6)
plt.xlabel('Apparent radius [degrees]')
#plt.ylabel('Absolute magnitude [mag]')
plt.locator_params(nbins=4)

plt.figure()
plt.subplot(2,2,1)
plt.scatter(vdispr, radiusr, color = 'k', alpha = 0.6)
#plt.xlabel('Whitened velocity dispersion')
plt.xlim([-0.05,0.05])
plt.ylabel('Whitened apparent radius')
plt.ylim([21,27])
plt.ylim([-12,12])
plt.locator_params(nbins=4)
plt.subplot(2,2,3)
plt.scatter(vdispr, absmr, color = 'k', alpha = 0.6)
plt.xlabel('Whitened velocity dispersion')
plt.xlim([-0.05,0.05])
plt.ylabel('Whitened absolute magnitude')
plt.locator_params(nbins=4)
plt.subplot(2,2,4)
plt.scatter(radiusr,absmr, color = 'k', alpha = 0.6)
plt.xlabel('Whitened apparent radius')
plt.xlim([-12,12])
#plt.ylabel('Whitened absolute magnitude')
plt.locator_params(nbins=4)

plt.figure()
plt.subplot(2,2,1)
plt.scatter(new3, new1, color = 'k', alpha = 0.6)
plt.ylim([-12,12])
plt.locator_params(nbins=4)
plt.subplot(2,2,3)
plt.scatter(new3, new2, color = 'k', alpha = 0.6)
plt.locator_params(nbins=4)
plt.subplot(2,2,4)
plt.scatter(new1, new2, color = 'k', alpha = 0.6)
plt.xlim([-12,12])
plt.locator_params(nbins=4)