#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>


int main(int argc,char** argv) {
    if(argc <= 1 || argc > 3) {
        printf("format:\n tex-gen \"albedo path\" ... {optional} \"normal map path\"\n");
        return 1;
    }

    FILE* file = fopen("out.bin","wb");
    if(argc == 2) { // only albedo
        printf("generating texture {albedo path: %s,normal map path: none} ... ",argv[1]);
        // TODO: add initial texture contents bytes
        // TODO: make a separate function for loading albedo bytes
        int alb_w,alb_h;
        unsigned char* data = stbi_load(argv[1],&alb_w,&alb_h,NULL,3);
        if(!data) {
            printf("aborted\n error reading albedo file\n");
            return 1;
        }
        fwrite(&alb_w,sizeof(int),1,file);
        fwrite(&alb_h,sizeof(int),1,file);
        fwrite(data,sizeof(unsigned char),alb_w*alb_h*3,file);

        fclose(file);
        printf("done\n");
    }
    else if(argc == 3) { // albedo + normal map

    }
    return 0;
}