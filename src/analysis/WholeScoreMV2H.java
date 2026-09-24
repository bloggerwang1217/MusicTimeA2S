import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import mv2h.Main;
import mv2h.objects.Music;
import mv2h.objects.Note;

/**
 * Uses one minimum-cost whole-score alignment with fixed diagonal/up/left ties.
 * Scoring and time warping use the existing MV2H classes unchanged.
 * This differs from Main -a, which maximizes the score over all optimal paths.
 */
public class WholeScoreMV2H {
    static int[][] pitches(List<List<Note>> chords) {
        int[][] result = new int[chords.size()][];
        for (int i=0;i<result.length;i++) {
            result[i]=new int[chords.get(i).size()];
            for(int j=0;j<result[i].length;j++) result[i][j]=chords.get(i).get(j).pitch;
            Arrays.sort(result[i]);
        }
        return result;
    }
    static double distance(int[] a,int[] b) {
        int i=0,j=0,matches=0;
        while(i<a.length && j<b.length) {
            if(a[i]==b[j]) {matches++;i++;j++;}
            else if(a[i]<b[j])i++;else j++;
        }
        return matches==0 ? 1.0 : 1.0-Main.getF1(matches,b.length-matches,a.length-matches);
    }
    public static void main(String[] args) throws Exception {
        Music gt=Music.parseMusic(new Scanner(new File(args[0])));
        Music pred=Music.parseMusic(new Scanner(new File(args[1])));
        int[][] g=pitches(gt.getNoteLists()),p=pitches(pred.getNoteLists());
        int n=g.length,m=p.length;
        System.err.println("chords="+n+","+m);
        if(n==0 || m==0) throw new IllegalArgumentException("Empty score: handle explicitly outside scorer");
        byte[][] pointers=new byte[n+1][m+1];
        double[] prev=new double[m+1];Arrays.fill(prev,Double.POSITIVE_INFINITY);prev[0]=0;
        for(int i=1;i<=n;i++) {
            double[] cur=new double[m+1];cur[0]=Double.POSITIVE_INFINITY;
            for(int j=1;j<=m;j++) {
                double best=prev[j-1]+distance(g[i-1],p[j-1]);byte direction=0;
                double up=prev[j]+1.0,left=cur[j-1]+1.0;
                if(up<best) {best=up;direction=-1;}
                if(left<best) {best=left;direction=1;}
                cur[j]=best;pointers[i][j]=direction;
            }
            prev=cur;
        }
        List<Integer> alignment=new ArrayList<>();
        int i=n,j=m;
        while(i>0 || j>0) {
            byte d=pointers[i][j];
            if(d==0) {alignment.add(j-1);i--;j--;}
            else if(d==-1) {alignment.add(-1);i--;}
            else {j--;}
        }
        Collections.reverse(alignment);
        if(alignment.size()!=n) throw new IllegalStateException("Wrong alignment length");
        System.err.println("cost="+prev[m]+" anchors="+alignment.stream().filter(x->x>=0).count());
        if(args.length>2) java.nio.file.Files.writeString(java.nio.file.Path.of(args[2]),alignment.toString());
        Main.ONSET_DELTA=0;Main.DURATION_DELTA=20;Main.GROUPING_EPSILON=20;
        System.out.println(gt.evaluateTranscription(pred.align(gt,alignment)));
    }
}
