#include "objects.cuh"
#include <stdio.h>

void handlePhysics(object* scene,const int& sceneSize) {
	for(int i = 0; i < sceneSize; i++) {
		for(int j = i+1; j < sceneSize; j++) {
			object &a = scene[i],&b = scene[j];
			if(a.sphere && b.sphere) {
				float dist = -((b.a - a.a).len()-a.b.x-b.b.x);
				if(dist > 0) {
					vec3 n = (a.a - b.a).norm();
					a.a = a.a + n * (dist / 2 + 0.001);
					b.a = b.a - n * (dist / 2 + 0.001);
					vec3 rv = b.velocity - a.velocity;
					float velAlongNorm = dot(rv,n);
					if(velAlongNorm > 0) return;
					float K = -1.5 * velAlongNorm;
					float aMass = a.b.x * a.b.x,bMass = b.b.x * b.b.x;
					K /= (1 / aMass + 1 / bMass);
					vec3 impulse = n * K;
					a.velocity = a.velocity - impulse / aMass;
					b.velocity = b.velocity + impulse / bMass;

				}
			}
			else if(a.sphere != b.sphere) {
				float dist = 0; vec3 p = {0,0,0};
				const int &idxTrig = (!scene[j].sphere) ? j : i;
				const int &idxSphere = scene[j].sphere ? j : i;
				trigSphereDist(idxSphere,idxTrig,dist,p,scene);
				if(dist < 0) {
					object &sphere=scene[idxSphere];
					vec3 n = p - sphere.a; // trig == a , sphere == b
					sphere.a=p-n.norm()*(sphere.b.x+0.001f);

					float velAlongNorm = dot(sphere.velocity,-n);
					if(velAlongNorm > 0) return;
					float K = -1.1 * velAlongNorm;
					float bMass = sphere.b.x * sphere.b.x;
					K /= (1 / bMass);
					vec3 impulse = n * K;
					sphere.velocity = -sphere.velocity + impulse / bMass;
				}
			}
		}
	}
}