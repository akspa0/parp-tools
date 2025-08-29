// docs/AlphaWDTReader/snippets/connected_components.cs
// Purpose: Reusable labeling for boolean occupancy grids (water masks).

using System.Collections.Generic;

namespace Snippets
{
    public static class Components
    {
        public static int[,] Label(bool[,] occ)
        {
            int w=occ.GetLength(0), h=occ.GetLength(1);
            var labels = new int[w,h];
            int id=0;
            var q = new Queue<(int x,int y)>();
            int[] dx={1,-1,0,0}, dy={0,0,1,-1};
            var seen = new bool[w,h];

            for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
            {
                if(!occ[x,y]||seen[x,y]) continue;
                id++; q.Enqueue((x,y)); seen[x,y]=true; labels[x,y]=id;
                while(q.Count>0)
                {
                    var (cx,cy)=q.Dequeue();
                    for(int k=0;k<4;k++)
                    {
                        int nx=cx+dx[k], ny=cy+dy[k];
                        if(nx<0||ny<0||nx>=w||ny>=h) continue;
                        if(!occ[nx,ny]||seen[nx,ny]) continue;
                        seen[nx,ny]=true; labels[nx,ny]=id; q.Enqueue((nx,ny));
                    }
                }
            }
            return labels;
        }
    }
}
