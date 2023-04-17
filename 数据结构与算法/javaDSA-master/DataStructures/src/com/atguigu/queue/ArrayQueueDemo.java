package com.atguigu.queue;

import java.util.Scanner;

public class ArrayQueueDemo {

	public static void main(String[] args) {
		//����
		ArrayQueue queue = new ArrayQueue(3);
		char key = ' ';//�����û�����
		java.util.Scanner scanner = new Scanner(System.in);
		boolean loop = true;
		
		//���һ���˵�
		while(loop) {
			System.out.println("s(show)����ʾ����");
			System.out.println("a(add)��������ݵ�����");
			System.out.println("g(get)���Ӷ����л�ȡԪ��");
			System.out.println("h(head)���鿴����Ͷ������");
			System.out.println("e(exit)���˳�����");
			key = scanner.next().charAt(0);//���ռ���������ַ���������ȡ�����ĵ�һ���ַ���
			
			switch(key) {
			case 's':
				queue.showQueue();
				break;
			case 'a':
				System.out.println("����һ������");
				int value = scanner.nextInt();
				queue.addQueue(value);
				break;
			case 'g': //���쳣�ˣ�Ҫ����
				try {
					int res = queue.getQueue();
					System.out.printf("ȡ����������%d\n",res);
				} catch (Exception e) {
					// ���������쳣�Ĵ���
					System.out.println(e.getMessage());
				}
				break;
			case 'h'://���쳣�ˣ���Ҫ������
				try {
					int res = queue.headQueue();
					System.out.printf("����ͷ��������%d\n",res);
				} catch (Exception e) {
					// ���������쳣�Ĵ���
					System.out.println(e.getMessage());
				}
				break;
			case 'e':
				scanner.close();
				loop = false;
				break;
			default:
				break;
			}
		}
		System.out.println("�����˳�");
	}
}

//ʹ������ģ����б�дһ��ArrayQueue��
class ArrayQueue{
	private int maxSize;//������������
	private int front;//����ͷ
	private int rear;//����β
	private int[] arr;//���ڴ������
	
	//�������еĹ�����
	public  ArrayQueue(int arrMaxSize) {
		maxSize = arrMaxSize;
		arr = new int[maxSize];
		front = -1;//ָ�����ͷ����������front��ָ�����ͷ��ǰһ��λ��
		rear = -1;//ָ�����β��ָ�����β�����ݣ����������һ�����ݣ�
	}
	
	//�ж϶����Ƿ�Ϊ��
	public boolean isEmpty() {
		return front == rear;
	}
		
	//�ж϶����Ƿ���
	public boolean isFull() {
		return rear == maxSize-1;
	}
		
	//������ݵ�����
	public void addQueue(int n) {
		if(isFull()) {
			System.out.println("�����������ܼ�������");
			return;
		}
		rear++;//βָ�����
		arr[rear]=n;
	}
		
	//��ȡ���е����ݣ�������
	public int getQueue() {
		//�п�
		if(isEmpty()) {
			//�׳��쳣
			throw new RuntimeException("���пգ�����ȡ����");
		}
		front++;//front����
		return arr[front];
	}
		
	//��ʾ���е���������
	public void showQueue() {
		//�п�
		if(isEmpty()) {
			System.out.println("���пգ�û�п���ʾ������");
			return;
		}
		//����
		for (int i = 0; i < arr.length; i++) {
			System.out.printf("arr[%d]=%d\n",i,arr[i]);
		}
	}
		
	//��ʾ���е�ͷ���ݣ�ע�ⲻ��ȡ��
	public int headQueue() {
		//�п�
		if(isEmpty()) {
			throw new RuntimeException("���пգ�û�п�ȡ��ͷ����");
		}
		return arr[front+1];//ע�⣺frontָ����Ƕ���ͷ��ǰһ�����ݡ�����С��һ��λ�ã�
	}
	
	
	
}
