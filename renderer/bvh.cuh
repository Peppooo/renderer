#include "scene.cuh"
#include <vector>
#include <iostream>

class box {
public:
	vec3 edge,size;
	int startIndex,length;
	__host__ __device__ bool contained(const vec3 v) {
		vec3 max = edge + size;
		return (v.x <= max.x && v.y <= max.y && v.z <= max.z && v.x >= edge.x && v.y >= edge.y && v.z >= edge.z);
	}
	__host__ __device__ int count_verts(const object obj) {
		int count = 0;
		for(int i = 0; i < 3; i++) {
			if(contained(obj[i])) {
				count++;
			}
		}
		return count;
	}
	__device__ bool intersect(const vec3& O,const vec3& D,float& dist) const {
		vec3 invDir = 1 / D;
		vec3 t0s = (edge - O) * invDir;
		vec3 t1s = (edge + size - O) * invDir;

		vec3 tmin3 = v_min(t0s,t1s);
		vec3 tmax3 = v_max(t0s,t1s);

		dist = max(max(tmin3.x,tmin3.y),tmin3.z);
		float tmax = min(min(tmax3.x,tmax3.y),tmax3.z);

		return (tmax >= max(dist,0.0f));
	}
};

class node {
public:
	box self;
	int child_nodeA = -1;
	int child_nodeB = -1;
	int depth = 0;
	__host__ __device__ int operator[](int i){
		if(i == 0) return child_nodeA;
		return child_nodeB;
	}
};

#define MAX_BVH_NODES 5000

class BVH {
	private:
		node* host_pointer;
		node* device_pointer;
		bool is_pointer_host;
	public:
		node* nodes;
		int nodesSize;
		__host__ void init() {
			nodes = new node[MAX_BVH_NODES];
			is_pointer_host = true;
			host_pointer = nodes;
			cudaMalloc(&device_pointer,sizeof(node) * MAX_BVH_NODES);
			nodesSize = 0;
		}
		__host__ void build(const int max_depth,object* scene,const size_t sceneSize) {
			nodesSize++;
			vec3 minb = {FLT_MAX,FLT_MAX,FLT_MAX};
			vec3 maxb = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
			for(int i = 0; i < sceneSize; i++) {
				minb = v_min(minb,v_min(v_min(scene[i].a,scene[i].b),scene[i].c));
				maxb = v_max(maxb,v_max(v_max(scene[i].a,scene[i].b),scene[i].c));
			}
			nodes[0].self = box();
			nodes[0].self.edge = minb;
			nodes[0].self.size = maxb - minb;
			nodes[0].depth = 0;
			nodes[0].self.length = sceneSize;
			nodes[0].self.startIndex = 0;
			buildChild(0,max_depth,scene,sceneSize);
		}
		__host__ void buildChild(const int idx,const int max_depth,object* scene,const size_t sceneSize) {
			if(nodesSize > MAX_BVH_NODES) {
				cout << "exceeded nodes size limit" << endl;
			}
			nodesSize++;
			nodes[idx].child_nodeA = nodesSize - 1;
			nodesSize++;
			nodes[idx].child_nodeB = nodesSize - 1;

			int left = nodes[idx].child_nodeA;
			int right = nodes[idx].child_nodeB;

			nodes[left].depth = nodes[idx].depth + 1;
			nodes[right].depth = nodes[idx].depth + 1;

			// calcolo bounding box dei figli
			vec3 minA = {FLT_MAX,FLT_MAX,FLT_MAX};
			vec3 maxA = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
			int axis = max_idx(nodes[idx].self.size);
			vector<int> index_a;
			vector<int> index_b;
			vector<int> middle;

			vec3 newLeftSize = nodes[idx].self.size;
			newLeftSize[axis] = (newLeftSize[axis]/2.0f);
			nodes[left].self.size = newLeftSize;
			nodes[left].self.edge = nodes[idx].self.edge;

			//float leftMax = (nodes[left].self.edge + nodes[left].self.size)[axis];
			index_a.clear(); index_b.clear(); middle.clear();
			//cout << "Depth: " << nodes[left].depth << " , " << endl;
			for(int i = nodes[idx].self.startIndex; i < (nodes[idx].self.startIndex+nodes[idx].self.length); i++) {
				int numVertsIn = nodes[left].self.count_verts(scene[i]);
				if(numVertsIn == 0) {
					index_b.push_back(i);
				}
				else if(numVertsIn == 1 || numVertsIn == 2) {
					//nodes[left].self.size[axis] = (v_max(scene[i].a,v_max(scene[i].b,scene[i].c))[axis] - nodes[left].self.edge[axis] + epsilon);
					middle.push_back(i);
				}
				else if(numVertsIn==3) {
					index_a.push_back(i);
				}
			}


			int leftChildLength = (index_a.size() + middle.size());
			int rightChildLength = (index_b.size() + middle.size());

			int leftChildStart = nodes[idx].self.startIndex;
			int rightChildStart = nodes[idx].self.startIndex + index_a.size();

			//cout << "depth: " << nodes[left].depth << " , node a length: " << leftChildLength << " , node b length: " << rightChildLength << endl;


			nodes[left].self.length = leftChildLength;    // assign positions of trigs contained in boundings
			nodes[left].self.startIndex = leftChildStart;

			nodes[right].self.length = rightChildLength;
			nodes[right].self.startIndex = rightChildStart;


			nodes[right].self.size = nodes[idx].self.size; // assign boundings size
			nodes[right].self.size[axis] -= nodes[left].self.size[axis];

			nodes[right].self.edge = nodes[idx].self.edge;
			nodes[right].self.edge[axis] += nodes[left].self.size[axis];

			object* new_scene = new object[sceneSize];

			copy(scene,scene+ sceneSize,new_scene);


			for (int i = 0; i < index_a.size(); i++) {
				new_scene[nodes[idx].self.startIndex + i] = scene[index_a[i]];
			}

			for (int i = 0; i < middle.size(); i++) {
				new_scene[nodes[idx].self.startIndex + index_a.size() + i] = scene[middle[i]];
			}

			for (int i = 0; i < index_b.size(); i++) {
				new_scene[nodes[idx].self.startIndex + i + index_a.size() + middle.size()] = scene[index_b[i]];
			}

			// Clean up memory after copying:

			//nodes[right].self.length = index_b.size();
			//nodes[right].self.startIndex = nodes[idx].self.startIndex + nodes[left].self.length;

			copy(new_scene,new_scene + sceneSize,scene);

			//cout << "child A trigs num: " << index_a.size() << " , B: " << index_b.size() << endl;

			if(nodes[left].depth < max_depth && nodes[left].self.length > 6) {
				buildChild(left,max_depth,scene,sceneSize);
			}
			if(nodes[right].depth < max_depth && nodes[right].self.length > 6) {
				buildChild(right,max_depth,scene,sceneSize);
			}

			cudaMemcpy(device_pointer,nodes,sizeof(node) * MAX_BVH_NODES,cudaMemcpyHostToDevice);
		}
		__device__ bool castRay(int idx,const vec3& o,const vec3& d,box& box) const {
			bool hit = false;
			float __dist;
			if(nodes[idx].self.intersect(o,d,__dist)) {
				if(nodes[idx].child_nodeA == -1 && nodes[idx].child_nodeB == -1) {
					get_back:
					box = nodes[idx].self;
					return true;
				}
				float c_dist = FLT_MAX;
				float min_dist = FLT_MAX;
				int hitIdx = -1;
				for(int j = 0; j < 2; j++) { // controllo quale dei figli colpisco ed e piu vicino
					if(nodes[idx][j] != -1) {
						if(nodes[nodes[idx][j]].self.intersect(o,d,c_dist) && c_dist < min_dist && c_dist > 0) {
							min_dist = c_dist;
							hitIdx = j;
						}
					}
				}
				if(hitIdx == -1) goto get_back;
				castRay(0,o,d,box);
			}
			return false;
		}
		void convertToDevice() {
			if(is_pointer_host) {
				nodes = device_pointer;
				is_pointer_host = false;
			}
		}
		void convertToHost() {
			if(!is_pointer_host) {
				nodes = host_pointer;
				is_pointer_host = true;
			}
		}
};