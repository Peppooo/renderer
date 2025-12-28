#include "scene.cuh"
#include <vector>
#include <iostream>

__device__ void d_swap(float& f1,float& f2) {
	float copy = f1;
	f1 = f2;
	f2 = copy;
}

class box {
public:
	vec3 Min,Max;
	int startIndex; int trigCount;
	vec3 center() const {
		return (Min + Max) * 0.5f;
	}
	__device__ bool intersect(const vec3& O,const vec3& D) const
	{
		float tmin = -FLT_MAX;
		float tmax = FLT_MAX;

		// X slab
		if(D.x != 0.0f)
		{
			float inv = 1.0f / D.x;
			float t0 = (Min.x - O.x) * inv;
			float t1 = (Max.x - O.x) * inv;
			if(t0 > t1) d_swap(t0,t1);
			tmin = max(tmin,t0);
			tmax = min(tmax,t1);
		}
		else if(O.x < Min.x || O.x > Max.x)
			return false;

		// Y slab
		if(D.y != 0.0f)
		{
			float inv = 1.0f / D.y;
			float t0 = (Min.y - O.y) * inv;
			float t1 = (Max.y - O.y) * inv;
			if(t0 > t1) d_swap(t0,t1);
			tmin = max(tmin,t0);
			tmax = min(tmax,t1);
		}
		else if(O.y < Min.y || O.y > Max.y)
			return false;

		// Z slab
		if(D.z != 0.0f)
		{
			float inv = 1.0f / D.z;
			float t0 = (Min.z - O.z) * inv;
			float t1 = (Max.z - O.z) * inv;
			if(t0 > t1) d_swap(t0,t1);
			tmin = max(tmin,t0);
			tmax = min(tmax,t1);
		}
		else if(O.z < Min.z || O.z > Max.z)
			return false;

		if(tmax < 0.0f || tmin > tmax)
			return false;

		return true;
	}
	void grow_to_include_point(const vec3& v) {
		Max = v_max(v,Max);
		Min = v_min(v,Min);
	}
	void grow_to_include(const object& t) {
		grow_to_include_point(t.a);
		grow_to_include_point(t.b);
		grow_to_include_point(t.c);
	}
};


class node {
public:
	box bounds;
	int leftChild = -1;
	int rightChild = -1;
	int depth = 0;
};
#define max_nodes 20000

class bvh {
private:
	node* dev_nodes;
public:
	node* nodes = new node[max_nodes];
	int nodesCount;
	void buildChilds(object* scene,const int sceneSize,const int idx,const int max_depth) {
		int left = nodesCount++;
		int right = nodesCount++;
		nodes[idx].leftChild = left;
		nodes[idx].rightChild = right;


		nodes[left].depth = nodes[idx].depth + 1;
		nodes[right].depth = nodes[idx].depth + 1;

		nodes[left].bounds.startIndex = nodes[idx].bounds.startIndex;
		nodes[left].bounds.trigCount = 0;
		nodes[left].bounds.Min = vec3::Zero;
		nodes[left].bounds.Max = vec3::Zero;

		nodes[right].bounds = nodes[left].bounds;

		int split_axis = max_idx(nodes[idx].bounds.Max - nodes[idx].bounds.Min);

		vec3 centerSum = vec3::Zero; float weightsSum = 0;
		for(int i = 0; i < nodes[idx].bounds.trigCount; i++) {
			float weight = scene[nodes[idx].bounds.startIndex + i].area();
			centerSum+=scene[nodes[idx].bounds.startIndex + i].center()*weight;
			weightsSum += weight;
		}
		vec3 parentCenter = centerSum / weightsSum;
		//vec3 parentCenter = nodes[idx].bounds.center();

		for(int i = 0; i < nodes[idx].bounds.trigCount; i++) {
			int current_idx = nodes[idx].bounds.startIndex + i;

			bool goLeftChild = parentCenter[split_axis] > (scene[current_idx].center())[split_axis];

			int split_side_idx = goLeftChild ? left : right;
			nodes[split_side_idx].bounds.grow_to_include(scene[current_idx]);

			if(goLeftChild) {
				int target_idx = nodes[split_side_idx].bounds.startIndex + nodes[split_side_idx].bounds.trigCount;
				swap(scene[current_idx],scene[target_idx]);

				nodes[right].bounds.startIndex++;
			}


			nodes[split_side_idx].bounds.trigCount++;

		}


		if(nodes[idx].depth < max_depth && nodesCount < (max_nodes - 3)) {
			if(nodes[left].bounds.trigCount > 6) {
				buildChilds(scene,sceneSize,left,max_depth);
			}
			if(nodes[right].bounds.trigCount > 6) {
				buildChilds(scene,sceneSize,right,max_depth);

			}

		}

	}
	void build(const int max_depth,object* scene,const int sceneSize) {
		nodesCount = 0;
		int root = nodesCount++;
		nodes[root].bounds.Min = vec3::Zero;
		nodes[root].bounds.Max = vec3::Zero;
		nodes[root].depth = 0;
		nodes[root].bounds.trigCount = 0;
		nodes[root].bounds.startIndex = 0;
		for(int i = 0; i < sceneSize; i++) {
			nodes[root].bounds.grow_to_include(scene[i]);
			nodes[root].bounds.trigCount++;
		}
		buildChilds(scene,sceneSize,root,max_depth);
		cudaMalloc(&dev_nodes,sizeof(node) * nodesCount);
		cudaMemcpy(dev_nodes,nodes,sizeof(node) * nodesCount,cudaMemcpyHostToDevice);

	}
	__device__ int castRayBox(const vec3& o,const vec3& d) {
		for(int i = 0; i < nodesCount; i++) {
			if(dev_nodes[i].leftChild == -1 && dev_nodes[i].rightChild == -1) {
				if(dev_nodes[i].bounds.intersect(o,d)) {
					return 1;
				}
			}
		}
		return -1;
	}
	__device__ int castRay(const Scene* scene,const vec3& o,const vec3& d,vec3& p,vec3& n) const {
		int stack[100]; int stackSize = 0;
		stack[stackSize++] = 0;
		float min_dist = FLT_MAX;
		float current_dist = 10000;
		int hitIdx = -1;
		// start from root
		while(stackSize > 0) {
			int current_idx = stack[stackSize - 1]; stackSize--;

			if(dev_nodes[current_idx].bounds.intersect(o,d)) {
				// if leaf node, add indecies
				if(dev_nodes[current_idx].leftChild == -1 && dev_nodes[current_idx].rightChild == -1) {
					// iterate triangles
					for(int i = 0; i < dev_nodes[current_idx].bounds.trigCount; i++) {
						vec3 _p,_n;
						if(scene->intersect(i + dev_nodes[current_idx].bounds.startIndex,o,d,_p,_n)) {
							current_dist = (_p - o).len2();
							if(current_dist < min_dist) {
								p = _p,n = _n;
								min_dist = current_dist;
								hitIdx = i + dev_nodes[current_idx].bounds.startIndex;
							}
						}
					}
				}
				else {
					// push children to stack
					if(dev_nodes[current_idx].leftChild != -1)
						stack[stackSize++]=(dev_nodes[current_idx].leftChild);
					if(dev_nodes[current_idx].rightChild != -1)
						stack[stackSize++]=(dev_nodes[current_idx].rightChild);
					if(stackSize > 100) {
						printf("STACK OUT OF BOUNDS\n");
					}
				}
			}
		}
		return hitIdx;
	}

	void printNodes()
	{
		int current_depth = 0;

		while(1) {
			cout << endl << " ----------------- DEPTH: " << current_depth << endl;
			int found = 0;
			for(int i = 0; i < nodesCount; i++) {
				if(nodes[i].depth == current_depth) {
					found++;
					cout << "Length: " << nodes[i].bounds.trigCount << " , childA,B: " << nodes[i].leftChild << " , " << nodes[i].rightChild << endl;
				}
			}
			if(found == 0) return;
			current_depth++;
		}
	}
};