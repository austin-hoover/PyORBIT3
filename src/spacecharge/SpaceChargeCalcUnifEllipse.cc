/////////////////////////////////////////////////////////////////////////////
//
// FILE NAME
//   SpaceChargeCalcUnifEllipse.cc
//
// AUTHOR
//    A. Shishlo
//
// Created:
//   11/09/10
//
// DESCRIPTION
//  This class calculates the space charge kicks for bunch. It represent the bunch as the set
//  of uniformly charged ellipses in the center of mass of the bunch system.
//  The space charge kick is transformed later into the lab system.
//
/////////////////////////////////////////////////////////////////////////////
#include "SpaceChargeCalcUnifEllipse.hh"
#include "BufferStore.hh"

#include "ParticleMacroSize.hh"

#include <iostream>
#include <cmath>
#include <cfloat>

using namespace OrbitUtils;

namespace {

const int MOMENT_COUNT = 10;

/** Diagonalizes a real symmetric 3x3 matrix with Jacobi rotations.
 *  The eigenvectors are returned as columns, so V^T A V is diagonal.
 */
void diagonalizeSymmetric3x3(double matrix[3][3], double eigenvalues[3], double eigenvectors[3][3])
{
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			eigenvectors[i][j] = (i == j) ? 1. : 0.;
		}
	}

	for(int iteration = 0; iteration < 50; iteration++){
		int p = 0;
		int q = 1;
		double max_off_diagonal = fabs(matrix[p][q]);
		if(fabs(matrix[0][2]) > max_off_diagonal){
			p = 0;
			q = 2;
			max_off_diagonal = fabs(matrix[p][q]);
		}
		if(fabs(matrix[1][2]) > max_off_diagonal){
			p = 1;
			q = 2;
			max_off_diagonal = fabs(matrix[p][q]);
		}

		double diagonal_scale = fabs(matrix[0][0]) + fabs(matrix[1][1]) + fabs(matrix[2][2]);
		if(max_off_diagonal <= 16.*DBL_EPSILON*diagonal_scale) break;

		double app = matrix[p][p];
		double aqq = matrix[q][q];
		double apq = matrix[p][q];
		double tau = (aqq - app)/(2.*apq);
		double t = ((tau >= 0.) ? 1. : -1.)/(fabs(tau) + hypot(1.,tau));
		double cosine = 1./sqrt(1. + t*t);
		double sine = t*cosine;

		matrix[p][p] = app - t*apq;
		matrix[q][q] = aqq + t*apq;
		matrix[p][q] = 0.;
		matrix[q][p] = 0.;
		for(int r = 0; r < 3; r++){
			if(r == p || r == q) continue;
			double arp = matrix[r][p];
			double arq = matrix[r][q];
			matrix[r][p] = cosine*arp - sine*arq;
			matrix[p][r] = matrix[r][p];
			matrix[r][q] = sine*arp + cosine*arq;
			matrix[q][r] = matrix[r][q];
		}

		for(int r = 0; r < 3; r++){
			double vrp = eigenvectors[r][p];
			double vrq = eigenvectors[r][q];
			eigenvectors[r][p] = cosine*vrp - sine*vrq;
			eigenvectors[r][q] = sine*vrp + cosine*vrq;
		}
	}

	for(int i = 0; i < 3; i++) eigenvalues[i] = matrix[i][i];
}

}

SpaceChargeCalcUnifEllipse::SpaceChargeCalcUnifEllipse(int nEllipses_in): CppPyWrapper(NULL)
{
	nEllipses = nEllipses_in;
  ellipsoidCalc_arr = new UniformEllipsoidFieldCalculator*[nEllipses];
	for(int ie = 0; ie < nEllipses; ie++){
		ellipsoidCalc_arr[ie] = new UniformEllipsoidFieldCalculator();
	}
	macroSizesEll_arr = (double* ) malloc (sizeof(double)*nEllipses);
	macroSizesEll_MPI_arr = (double* ) malloc (sizeof(double)*nEllipses);
	for(int ie = 0; ie < nEllipses; ie++){
		macroSizesEll_arr[ie] = 0.;
		macroSizesEll_MPI_arr[ie] = 0.;
	}
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			principalAxis_arr[i][j] = (i == j) ? 1. : 0.;
		}
	}
}


SpaceChargeCalcUnifEllipse::~SpaceChargeCalcUnifEllipse(){
	for(int ie = 0; ie < nEllipses; ie++){
		if(ellipsoidCalc_arr[ie]->getPyWrapper() != NULL){
			Py_DECREF(ellipsoidCalc_arr[ie]->getPyWrapper());
		} else {
			delete ellipsoidCalc_arr[ie];
		}
	}
	delete [] ellipsoidCalc_arr;

	free(macroSizesEll_arr);
	free(macroSizesEll_MPI_arr);
}


void SpaceChargeCalcUnifEllipse::trackBunch(Bunch* bunch, double length){

	int nPartsGlobal = bunch->getSizeGlobal();
	if(nPartsGlobal < 3) return;

	SyncPart* syncPart = bunch->getSyncPart();
	double beta = syncPart->getBeta();
	double gamma = syncPart->getGamma();

	for(int ie = 0; ie < nEllipses; ie++){
		ellipsoidCalc_arr[ie]->setQ(0.);
	}

	//analyse the bunch and make the ellipsoid filed sources
	this->bunchAnalysis(bunch);

	//if there is nothing we give up
	if(total_macrosize == 0.) return;

	double trans_factor =  length*bunch->getClassicalRadius()/(pow(beta,2)*pow(gamma,2));
	double long_factor =  length*bunch->getClassicalRadius()*bunch->getMass();

	double x,y,z,ex,ey,ez;
	for (int i = 0, n = bunch->getSize(); i < n; i++){
		x = bunch->x(i) - x_center;
		y = bunch->y(i) - y_center;
		z = (bunch->z(i) - z_center)*gamma;
		this->calculateField(x,y,z,ex,ey,ez);
		//calculate momentum kicks
		bunch->xp(i) += ex * trans_factor;
		bunch->yp(i) += ey * trans_factor;
		bunch->dE(i) += ez * long_factor;
	}
}

/** Analyses the bunch and sets up the ellipsoid filed sources */
void SpaceChargeCalcUnifEllipse::bunchAnalysis(Bunch* bunch){

	//Weighted raw moments: x,y,z,x2,y2,z2,xy,xz,yz and total macrosize.
	int buff_index0 = 0;
	int buff_index1 = 0;
	double* coord_avg = BufferStore::getBufferStore()->getFreeDoubleArr(buff_index0,MOMENT_COUNT);
	double* coord_avg_out = BufferStore::getBufferStore()->getFreeDoubleArr(buff_index1,MOMENT_COUNT);
	for (int i = 0; i < MOMENT_COUNT; i++){
		coord_avg[i] = 0.;
	}

	//Calculate local moments.
	bunch->compress();
	double** partArr = bunch->coordArr();
	double* coordArr = NULL;
	int has_msize = bunch->hasParticleAttributes("macrosize");
	if(has_msize > 0){
		ParticleMacroSize* macroSizeAttr = (ParticleMacroSize*) bunch->getParticleAttributes("macrosize");
		double m_size = 0.;
		for(int ip = 0, n = bunch->getSize(); ip < n; ip++){
			m_size = macroSizeAttr->macrosize(ip);
			coordArr = partArr[ip];
			coord_avg[0] += m_size*coordArr[0];
			coord_avg[1] += m_size*coordArr[2];
			coord_avg[2] += m_size*coordArr[4];
			coord_avg[3] += m_size*coordArr[0]*coordArr[0];
			coord_avg[4] += m_size*coordArr[2]*coordArr[2];
			coord_avg[5] += m_size*coordArr[4]*coordArr[4];
			coord_avg[6] += m_size*coordArr[0]*coordArr[2];
			coord_avg[7] += m_size*coordArr[0]*coordArr[4];
			coord_avg[8] += m_size*coordArr[2]*coordArr[4];
			coord_avg[9] += m_size;
		}
	} else {
		double m_size = bunch->getMacroSize();
		int nParts = bunch->getSize();
		coord_avg[9] = m_size*nParts;
		for(int ip = 0; ip < nParts; ip++){
			coordArr = partArr[ip];
			coord_avg[0] += coordArr[0];
			coord_avg[1] += coordArr[2];
			coord_avg[2] += coordArr[4];
			coord_avg[3] += coordArr[0]*coordArr[0];
			coord_avg[4] += coordArr[2]*coordArr[2];
			coord_avg[5] += coordArr[4]*coordArr[4];
			coord_avg[6] += coordArr[0]*coordArr[2];
			coord_avg[7] += coordArr[0]*coordArr[4];
			coord_avg[8] += coordArr[2]*coordArr[4];
		}
		for (int i = 0; i < 9; i++){
			coord_avg[i] *= m_size;
		}
	}

	//calculates sum over all  CPUs
	ORBIT_MPI_Allreduce(coord_avg,coord_avg_out,MOMENT_COUNT,MPI_DOUBLE,MPI_SUM,bunch->getMPI_Comm_Local()->comm);

	total_macrosize = coord_avg_out[9];
	if(total_macrosize == 0.){
	  //free resources
	  OrbitUtils::BufferStore::getBufferStore()->setUnusedDoubleArr(buff_index0);
	  OrbitUtils::BufferStore::getBufferStore()->setUnusedDoubleArr(buff_index1);
		return;
	}

	//Calculate the covariance in bunch-rest coordinates (x,y,gamma*z), then
	//rotate it to its principal axes. A uniform ellipsoid has <u_i^2>=a_i^2/5.
	x_center = coord_avg_out[0]/total_macrosize;
	y_center = coord_avg_out[1]/total_macrosize;
	z_center = coord_avg_out[2]/total_macrosize;
	double gamma = bunch->getSyncPart()->getGamma();
	double covariance[3][3];
	covariance[0][0] = coord_avg_out[3]/total_macrosize - x_center*x_center;
	covariance[1][1] = coord_avg_out[4]/total_macrosize - y_center*y_center;
	covariance[2][2] = (coord_avg_out[5]/total_macrosize - z_center*z_center)*gamma*gamma;
	covariance[0][1] = covariance[1][0] = coord_avg_out[6]/total_macrosize - x_center*y_center;
	covariance[0][2] = covariance[2][0] = (coord_avg_out[7]/total_macrosize - x_center*z_center)*gamma;
	covariance[1][2] = covariance[2][1] = (coord_avg_out[8]/total_macrosize - y_center*z_center)*gamma;
	double principal_variance[3];
	diagonalizeSymmetric3x3(covariance,principal_variance,principalAxis_arr);
	x2_avg = fabs(principal_variance[0]);
	y2_avg = fabs(principal_variance[1]);
	z2_avg = fabs(principal_variance[2]);
	a2_ellips = 5.0*x2_avg;
	b2_ellips = 5.0*y2_avg;
	c2_ellips = 5.0*z2_avg;
	a_ellips = sqrt(a2_ellips);
	b_ellips = sqrt(b2_ellips);
	c_ellips = sqrt(c2_ellips);

	//std::cout<<"debug a_ellips="<< a_ellips <<" b_ellips="<< b_ellips <<" c_ellips="<< c_ellips <<std::endl;
	//free resources
	OrbitUtils::BufferStore::getBufferStore()->setUnusedDoubleArr(buff_index0);
	OrbitUtils::BufferStore::getBufferStore()->setUnusedDoubleArr(buff_index1);

	//check if the beam size is not zero
  if( x2_avg == 0. || y2_avg == 0.|| z2_avg == 0.){
		int rank = 0;
		ORBIT_MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(rank == 0){
			std::cerr << "SpaceChargeCalcUnifEllipse::bunchAnalysis(bunch,...)" << std::endl
         				<< "The bunch coords min and max sizes are wrong! Cannot calculate space charge!" << std::endl
								<<" x2_rms="<< x2_avg << std::endl
								<<" y2_rms="<< y2_avg << std::endl
								<<" z2_rms="<< z2_avg << std::endl
								<< "Stop."<< std::endl;
		}
		ORBIT_MPI_Finalize();
  }

	//if we have only one ellipse we should not distribute anything
	if(nEllipses == 1){
	  macroSizesEll_arr[0] = total_macrosize;
		double r_max = a_ellips;
		if(r_max < b_ellips) r_max = b_ellips;
		if(r_max < c_ellips) r_max = c_ellips;
    ellipsoidCalc_arr[0]->setEllipsoid(a_ellips,b_ellips,c_ellips,10.*r_max);
		ellipsoidCalc_arr[0]->setQ(macroSizesEll_arr[0]);
		return;
	}

	//find the distribution of the macrosizes between nEllipses
	for(int ie = 0; ie < nEllipses; ie++){
		macroSizesEll_arr[ie] = 0.;
	}

	double pos = 0.;
	int pos_index = 0;
	ParticleMacroSize* macroSizeAttr = NULL;
	if(has_msize > 0){
		macroSizeAttr = (ParticleMacroSize*) bunch->getParticleAttributes("macrosize");
	}
	for(int ip = 0, n = bunch->getSize(); ip < n; ip++){
		coordArr = partArr[ip];
		double rest_coord[3] = {
			coordArr[0] - x_center,
			coordArr[2] - y_center,
			(coordArr[4] - z_center)*gamma
		};
		double principal_coord[3] = {0.,0.,0.};
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				principal_coord[i] += principalAxis_arr[j][i]*rest_coord[j];
			}
		}
		pos = sqrt(principal_coord[0]*principal_coord[0]/a2_ellips
		         + principal_coord[1]*principal_coord[1]/b2_ellips
		         + principal_coord[2]*principal_coord[2]/c2_ellips);
		pos_index = int(pos*nEllipses) - 1;
		if(pos_index < 0) pos_index = 0;
		if(pos_index >= nEllipses) pos_index = nEllipses - 1;
		double m_size = bunch->getMacroSize();
		if(macroSizeAttr != NULL) m_size = macroSizeAttr->macrosize(ip);
		macroSizesEll_arr[pos_index] += m_size;
	}
	//calculates sum over all  CPUs
	ORBIT_MPI_Allreduce(macroSizesEll_arr,macroSizesEll_MPI_arr,nEllipses,MPI_DOUBLE,MPI_SUM,bunch->getMPI_Comm_Local()->comm);
	for(int ie = 0; ie < nEllipses; ie++){
		macroSizesEll_arr[ie] = macroSizesEll_MPI_arr[ie];
		//std::cout<<"debug 0 ie ="<< ie <<" macrosize="<< macroSizesEll_MPI_arr[ie] << std::endl;
	}
	//calculate the relative volume density in each region. This density is a sum of all elipsoids
	for(int ie = 0; ie < nEllipses; ie++){
		macroSizesEll_MPI_arr[ie] /= ((ie+2)*(ie+2)*(ie+2) - (ie+1)*(ie+1)*(ie+1));
		//std::cout<<"debug 1 ie ="<< ie <<" macrosize="<< macroSizesEll_MPI_arr[ie] << std::endl;
	}
	//calculate the density for each elipsoid
	double rho_sum = 0.;
	for(int ie = (nEllipses-1); ie >= 0; ie--){
		macroSizesEll_MPI_arr[ie] -= rho_sum;
		rho_sum += macroSizesEll_MPI_arr[ie];
		//std::cout<<"debug 2 ie ="<< ie <<" macrosize="<< macroSizesEll_MPI_arr[ie] << " rho_sum="<< rho_sum <<std::endl;
	}

	//now set up the relative total charges in ellipsoids
	double q_sum = 0.;
	for(int ie = 0; ie < nEllipses; ie++){
		macroSizesEll_MPI_arr[ie] = macroSizesEll_MPI_arr[ie]*(ie+1)*(ie+1)*(ie+1);
		//std::cout<<"debug 3 ie ="<< ie <<" macrosize="<< macroSizesEll_MPI_arr[ie] << std::endl;
		q_sum += macroSizesEll_MPI_arr[ie];
	}
	double q_coeff = total_macrosize/q_sum;
	for(int ie = 0; ie < nEllipses; ie++){
		macroSizesEll_arr[ie] = macroSizesEll_MPI_arr[ie]*q_coeff;
	}

	//now we initialize the ellipses filed calculators
	double r_max = a_ellips;
	if(r_max < b_ellips) r_max = b_ellips;
	if(r_max < c_ellips) r_max = c_ellips;
	for(int ie = 0; ie < nEllipses; ie++){
		double coeff = (ie+2.)/nEllipses;
		ellipsoidCalc_arr[ie]->setEllipsoid(a_ellips*coeff,b_ellips*coeff,c_ellips*coeff,10.*r_max*coeff);
		ellipsoidCalc_arr[ie]->setQ(macroSizesEll_arr[ie]);
		//std::cout<<"debug ie ="<< ie <<" macrosize="<< macroSizesEll_arr[ie]<<" a="
		//         << a_ellips*coeff<< " b="<< b_ellips*coeff<< "  c="<< c_ellips*coeff*gamma
		//				 << " r_max="<< 10.*r_max*coeff << std::endl;
	}
}

/** Calculates the electric filed in the center of the bunch sytem. */
void SpaceChargeCalcUnifEllipse::calculateField(double x,  double y,  double z,
	                                            double& ex, double& ey, double& ez)
{
	ex = 0.;  ey = 0.; ez = 0.;
	double rest_coord[3] = {x,y,z};
	double principal_coord[3] = {0.,0.,0.};
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			principal_coord[i] += principalAxis_arr[j][i]*rest_coord[j];
		}
	}
	double x2 = principal_coord[0]*principal_coord[0];
	double y2 = principal_coord[1]*principal_coord[1];
	double z2 = principal_coord[2]*principal_coord[2];
	double ex_l,ey_l,ez_l;
	double principal_field[3] = {0.,0.,0.};
	for(int ie = 0; ie < nEllipses; ie++){
		ellipsoidCalc_arr[ie]->calcField(principal_coord[0],principal_coord[1],principal_coord[2],x2,y2,z2,ex_l,ey_l,ez_l);
		//std::cout<<"debug ie="<<ie<<" ex_l="<<ex_l<<" ey_l="<<ey_l<<" ez_l="<<ez_l<<std::endl;
		principal_field[0] += ex_l;
		principal_field[1] += ey_l;
		principal_field[2] += ez_l;
	}
	for(int i = 0; i < 3; i++){
		ex += principalAxis_arr[0][i]*principal_field[i];
		ey += principalAxis_arr[1][i]*principal_field[i];
		ez += principalAxis_arr[2][i]*principal_field[i];
	}
}

/** Returns the UniformEllipsoidFieldCalculator class instance with a particular index */
UniformEllipsoidFieldCalculator* SpaceChargeCalcUnifEllipse::getEllipsFieldCalculator(int ellipse_index)
{
	if(ellipse_index >= 0 && ellipse_index < nEllipses){
		return ellipsoidCalc_arr[ellipse_index];
	} else {
		return NULL;
	}
}

/** Returns the number of UniformEllipsoidFieldCalculator class instances */
int SpaceChargeCalcUnifEllipse::getNEllipses()
{
	return nEllipses;
}
