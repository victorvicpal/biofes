import matplotlib.pyplot as plt
import numpy
import pandas
import ClassicBip
import CanonicalBip

class plot(object):
	''' 
	Plot functions for CanonicalBip and ClassicBip
	'''

	def __init__(self, bip, target = None, dim = [0, 1] , xlim=[-2,2] , ylim= [-2,2], figx = 10, figy = 10, arrow_width = 0.1, font_size = 12, radius = 'Bonf'):
		if isinstance(bip, ClassicBip.ClassicBip):
			__type__ = "Classic"
		elif isinstance(bip, CanonicalBip.CanonicalBip):
			__type__ = "Canonical"
		else:
			raise ValueError('Undefined biplotpy class')

		if __type__ == 'Classic':
			if isinstance(target, (numpy.ndarray, list, pandas.core.series.Series)):
				fig = plt.figure(figsize=(figx,figy))
				ax1 = fig.add_subplot(111)

				ax1.scatter(bip.RowCoord[:, dim[0]], bip.RowCoord[:, dim[1]], c = target)
				for i in range(0,bip.ColCoord.shape[0]):
					ax1.arrow(0 ,0 , bip.ColCoord[i, dim[0]],
						bip.ColCoord[i, dim[1]], width = arrow_width )
					ax1.text(bip.ColCoord[i ,dim[0]], 
						bip.ColCoord[i,dim[1]] , bip.col_names[i], fontsize=font_size)

				plt.show()
			else:
				fig = plt.figure(figsize=(figx,figy))
				ax1 = fig.add_subplot(111)

				ax1.scatter(bip.RowCoord[:, dim[0]], bip.RowCoord[:, dim[1]])
				for i in range(0,bip.ColCoord.shape[0]):
					ax1.arrow(0 ,0 , bip.ColCoord[i, dim[0]],
						bip.ColCoord[i, dim[1]], width = arrow_width )
					ax1.text(bip.ColCoord[i ,dim[0]], 
						bip.ColCoord[i,dim[1]] , bip.col_names[i], fontsize=font_size)

				plt.show()
		elif __type__ == "Canonical":
			circles = []
			for i, el in enumerate(bip.GroupNames):
				circles.append(plt.Circle(bip.Group_Coord[i, dim], bip.Radius[radius][i], alpha = 0.4))

			fig, ax = plt.subplots(figsize=(figx,figy))

			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

			for i, circle in enumerate(circles):
				ax.add_artist(circle)
				ax.text(bip.Group_Coord[i, dim[0]],
					bip.Group_Coord[i, dim[1]],  bip.GroupNames[i])

			for i in range(0, bip.Var_Coord.shape[0]):
				ax.arrow(0,0, bip.Var_Coord[i, dim[0]], bip.Var_Coord[i, dim[1]], width= arrow_width)
				ax.text(bip.Var_Coord[i ,dim[0]], 
					bip.Var_Coord[i,dim[1]] , bip.col_names[i], fontsize=font_size)

			plt.show()

