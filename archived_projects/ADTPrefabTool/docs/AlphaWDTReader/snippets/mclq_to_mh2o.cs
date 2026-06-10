// docs/AlphaWDTReader/snippets/mclq_to_mh2o.cs
// Purpose: Minimal MCLQ → MH2O conversion via connected components per chunk.
// References: noggit3 liquid_layer/liquid_chunk, MapUpconverter MH2O structs.

using System;
using System.Collections.Generic;

namespace Snippets
{
    public static class MclqToMh2o
    {
        public sealed class Component
        {
            public int MinX, MinY, MaxX, MaxY;
            public List<(int x, int y)> Cells = new();
        }

        public static List<Component> FindComponents(bool[,] occupied)
        {
            int w = occupied.GetLength(0), h = occupied.GetLength(1);
            var visited = new bool[w, h];
            var result = new List<Component>();
            var q = new Queue<(int x,int y)>();

            int[] dx = {1,-1,0,0};
            int[] dy = {0,0,1,-1};

            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (!occupied[x,y] || visited[x,y]) continue;
                var comp = new Component{MinX=x,MinY=y,MaxX=x,MaxY=y};
                q.Enqueue((x,y)); visited[x,y] = true;
                while (q.Count>0)
                {
                    var (cx,cy)=q.Dequeue();
                    comp.Cells.Add((cx,cy));
                    comp.MinX = Math.Min(comp.MinX,cx);
                    comp.MinY = Math.Min(comp.MinY,cy);
                    comp.MaxX = Math.Max(comp.MaxX,cx);
                    comp.MaxY = Math.Max(comp.MaxY,cy);
                    for (int k=0;k<4;k++)
                    {
                        int nx=cx+dx[k], ny=cy+dy[k];
                        if (nx<0||ny<0||nx>=w||ny>=h) continue;
                        if (!occupied[nx,ny]||visited[nx,ny]) continue;
                        visited[nx,ny]=true; q.Enqueue((nx,ny));
                    }
                }
                result.Add(comp);
            }
            return result;
        }
    }
}
