#include "algebra.cuh"

enum material_type {
	diffuse,
	specular,
	glossy
};

class material {
public:
	material_type type;
	float glossiness;
	material() {};
	material(material_type Type,float Glossiness = 0): type(Type),glossiness(Glossiness) {};
	__device__ __forceinline__ vec3 bounce(const vec3& D,const vec3& n,curandStatePhilox4_32_10_t* state) {
		if(type == specular) {
			return D-n*2*dot(D,n);
		}
		else if(type == glossy) {
			vec3 specular_dir = D - n * 2 * dot(D,n);

			float weights[5];
			randomList(weights,5,state);
			weights[4] *= 4; // account for all the 4 vectors im going to sum up
			float _sum = sum(weights,5);

			vec3 lat1 = {-specular_dir.y,specular_dir.x,0};
			vec3 lat2 = cross(specular_dir,lat1);

			lat1 = (lat1 * glossiness + specular_dir * (1 - glossiness));
			lat2 = (lat2 * glossiness + specular_dir * (1 - glossiness));
			vec3 lat3 = ((-lat1) * glossiness + specular_dir * (1 - glossiness));
			vec3 lat4 = ((-lat2) * glossiness + specular_dir * (1 - glossiness));

			
			return ((lat1.norm() * weights[0] + lat2.norm()*weights[1]+lat3.norm()*weights[2]+lat4.norm()*weights[3]+specular_dir.norm()*weights[4]) / _sum);
		}
	}
};