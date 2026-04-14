
typedef std::pair<double, double> Vec2d;

/*
	Compute vorticity feature at cell coordinate (x, y) of the field f
*/
float vorticity(int x, int y, int width, int height, float* f){
    float dx=0.01;
    float dy=0.01;

    uint32_t idx = y*width+x; 
    
    int start_x = (x == 0) ? 0 : x - 1;
    int end_x = (x == width-1) ? x : x + 1;

    int start_y = (y == 0) ? 0 : y - 1;
    int end_y = (y == height-1) ? y : y + 1;

    uint32_t duidx = (start_y*width + end_x)*2;
    uint32_t dvidx = (end_y*width + start_x)*2;

    Vec2d fdu(f[duidx],f[duidx+1]);
    Vec2d fdv(f[dvidx],f[dvidx+1]);
    Vec2d vec0(f[idx*2],f[idx*2+1]);

    float duy = (fdu.second - vec0.second)/(dx*(end_x-start_x));
    float dvx = (fdv.first - vec0.first)/(dy*(end_y-start_y));

    return duy-dvx;
}
