package com.atguigu.sparsearray;

public class SparseArray {

	public static void main(String[] args) {
		//����һ��ԭʼ�Ķ�ά����11*11
		//0��û�����ӣ�1��ʾ���ӣ�2��ʾ����
		int chessArr1[][] = new int [11][11];
		chessArr1[1][2] = 1;
		chessArr1[2][3] = 2;
		chessArr1[4][5] = 2;
		
		//���ԭʼ�Ķ�ά����
		System.out.println("ԭʼ�Ķ�ά���飺");
		for (int [] row : chessArr1) {
			for(int data: row) {
				System.out.printf("%d\t", data);
			}	
			System.out.println();
		}
	
		//����ά���� ת ϡ������
		//1���ȱ�����ά���飬�õ���0���ݵĸ���
		int sum = 0;
		for(int i=0; i<chessArr1.length; i++) {
			for(int j=0; j<chessArr1[0].length; j++) {
				if(chessArr1[i][j] != 0)
					sum++;
			}
		}
		System.out.println("sum="+sum);
		System.out.println("chessArr1.length="+chessArr1.length);
		System.out.println("chessArr1[0].length="+chessArr1[0].length);
		

		//2��������Ӧ��ϡ������
		int sparseArr[][] = new int[sum+1][3];
		//��ϡ�����鸳ֵ
		sparseArr[0][0] = chessArr1.length;
		System.out.println("sparseArr[0][0]="+sparseArr[0][0]);
		sparseArr[0][1] = chessArr1[0].length;
		sparseArr[0][2] = sum;
		//������ά���飬����0��ֵ��ŵ�ϡ�������С�


		int count = 0; //���ڼ�¼�ǵڼ�����0����
		for(int i=0; i<chessArr1.length; i++) {
			for(int j=0; j<chessArr1[0].length; j++) {
				if(chessArr1[i][j] != 0) {
					count++;
					sparseArr[count][0] = i;
					sparseArr[count][1] = j;
					sparseArr[count][2] = chessArr1[i][j];
				}
				
			}
		}
		
		
		//���ϡ���������ʽ
		System.out.println();
		System.out.println("�õ���ϡ������Ϊ��");
		for(int i=0; i<sparseArr.length;i++) {
			System.out.printf("%d\t%d\t%d\t\n", sparseArr[i][0], sparseArr[i][1], sparseArr[i][2]);
		}
		System.out.println();
		
		//��ϡ������ ת ��ά����
		//1���ȶ�ȡϡ������ĵ�һ�У����ݵ�һ�е����ݣ�����ԭʼ�Ķ�ά����
		int chessArr2[][] =new int[sparseArr[0][0]][ sparseArr[0][1] ];
		
		//2���ٶ�ȡϡ��������е����飬������ԭʼ�Ķ�ά���鼴�ɡ�
		for (int i = 1; i < sparseArr.length; i++) {
			chessArr2[sparseArr[i][0]][sparseArr[i][1]] = sparseArr[i][2];
		}
		//�����ԭ��Ķ�ά����

		for(int[] row : chessArr2) {
			for(int data : row) {
				System.out.printf("%d\t", data);
			}
			System.out.println();
		}
	}

}
