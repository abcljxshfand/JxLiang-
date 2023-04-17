package com.atguigu.queue;

import java.util.Scanner;

public class CircleArrayQueueDemo {

	public static void main(String[] args) {
		//����
		ArrayQueue queue = new ArrayQueue(4);//ע�⣺��һ���յ�λ�ã�ʵ���Ϊ3
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
class CircleArray{
	private int maxSize;//������������
	private int front;
		//����ͷ������Ϊָ����еĵ�һ��Ԫ�أ���arr[front]����ʼֵ��Ϊ0
	private int rear;
		//����β������Ϊָ��������һ��Ԫ�صĺ�һ��λ�ã����ճ�һ���ռ���ΪԼ������ʼֵ��Ϊ0
	private int[] arr;//���ڴ������
	
	//�������еĹ�����
	public  CircleArray(int arrMaxSize) {
		maxSize = arrMaxSize;
		arr = new int[maxSize];
	}
	
	//�ж϶����Ƿ�Ϊ��
	public boolean isEmpty() {
		return front == rear;
	}
		
	//�ж϶����Ƿ���
	public boolean isFull() {
		return (rear+1)%maxSize == front;
	}
		
	//������ݵ�����
	public void addQueue(int n) {
		if(isFull()) {
			System.out.println("�����������ܼ�������");
			return;
		}
		arr[rear]=n; //rearָ�����һ��Ԫ�صĺ�һ������ֱ�����
		rear=(rear+1)%maxSize;//���ǻ��Σ�ȡģ
	}
		
	//��ȡ���е����ݣ�������
	public int getQueue() {
		//�п�
		if(isEmpty()) {
			//�׳��쳣
			throw new RuntimeException("���пգ�����ȡ����");
		}
		/*
		 * frontָ����еĵ�һ��Ԫ��
		 * 1���Ȱ�front��Ӧ��ֵ������һ����ʱ����
		 * 2����front���ƣ�����ȡģ
		 * 3������ʱ����ı�������
		 * */
		int value = arr[front];
		front = (front+1)%maxSize;
		return value;
	}
		
	//��ʾ���е���������
	public void showQueue() {
		//�п�
		if(isEmpty()) {
			System.out.println("���пգ�û�п���ʾ������");
			return;
		}
		//����
		//ע�⣺��front��ʼ�������������ٸ�Ԫ�أ�Ԫ�ظ�����ô��
		for (int i = front; i < front+size(); i++) {
			System.out.printf("arr[%d]=%d\n", i%maxSize, arr[i%maxSize]);
		}
	}
	
	//��ǰ������Ч���ݵĸ���
	public int size() {
		return (rear+maxSize-front)%maxSize;
	}
		
	//��ʾ���е�ͷ���ݣ�ע�ⲻ��ȡ��
	public int headQueue() {
		//�п�
		if(isEmpty()) {
			throw new RuntimeException("���пգ�û�п�ȡ��ͷ����");
		}
		return arr[front];
			//ע�⣺frontָ����Ƕ���ͷ������frontʱ�Ѿ�����ȡģ�Ķ�����front��Ϊ��ȷ��λ�á�
	}
	
	
	
}
